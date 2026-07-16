"""PyTorch backend for the tiny modular-addition transformer."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, cast

import torch  # type: ignore[import-not-found]
from torch import nn

from verifier_bottleneck.data.modular_addition import (
    ModularAdditionExample,
    make_modular_addition_split,
)
from verifier_bottleneck.experiments.modular_addition import (
    SEQUENCE_LENGTH,
    ExperimentConfig,
)
from verifier_bottleneck.verifiers.noisy import measure_noisy_verifier


class TinyCausalTransformer(nn.Module):
    """A small decoder-style transformer that predicts the answer token."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        modulus = config.task.modulus
        model = config.model
        self.token_embedding = nn.Embedding(modulus + 2, model.d_model)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, model.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=model.d_model,
            nhead=model.heads,
            dim_feedforward=model.d_ff,
            dropout=model.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=model.layers,
            norm=nn.LayerNorm(model.d_model),
        )
        self.output = nn.Linear(model.d_model, modulus)
        causal_mask = torch.triu(
            torch.ones(SEQUENCE_LENGTH, SEQUENCE_LENGTH, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, tokens: Any) -> Any:
        """Return logits for the equation answer."""
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.transformer(hidden, mask=self.causal_mask)
        return self.output(hidden[:, -1, :])


def _examples_to_tensors(
    examples: Sequence[ModularAdditionExample],
    *,
    modulus: int,
) -> tuple[Any, Any]:
    tokens = torch.tensor(
        [example.input_tokens(modulus) for example in examples],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [example.target for example in examples],
        dtype=torch.long,
    )
    return tokens, targets


def _evaluate(
    model: TinyCausalTransformer,
    tokens: Any,
    targets: Any,
) -> tuple[float, float, list[bool]]:
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        loss = nn.functional.cross_entropy(logits, targets).item()
        predictions = logits.argmax(dim=-1)
        correctness_tensor = predictions.eq(targets)
        accuracy = correctness_tensor.float().mean().item()
    correctness = [bool(value) for value in correctness_tensor.cpu().tolist()]
    return float(loss), float(accuracy), correctness


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _select_device(requested: str) -> Any:
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "device=cuda was requested, but torch.cuda.is_available() is false. "
            "Enable a GPU runtime or use device=cpu."
        )
    if requested == "auto":
        requested = "cuda" if cuda_available else "cpu"
    return torch.device(requested)


def train_modular_addition(
    config: ExperimentConfig,
    *,
    device: str,
) -> dict[str, object]:
    """Train and evaluate the configured transformer."""
    config.validate()
    _set_seed(config.seed)
    selected_device = _select_device(device)
    split = make_modular_addition_split(
        modulus=config.task.modulus,
        train_fraction=config.task.train_fraction,
        seed=config.seed,
    )
    train_tokens_cpu, train_targets_cpu = _examples_to_tensors(
        split.train,
        modulus=split.modulus,
    )
    test_tokens, test_targets = _examples_to_tensors(
        split.test,
        modulus=split.modulus,
    )
    train_tokens = train_tokens_cpu.to(selected_device)
    train_targets = train_targets_cpu.to(selected_device)
    test_tokens = test_tokens.to(selected_device)
    test_targets = test_targets.to(selected_device)

    model = TinyCausalTransformer(config).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    batch_size = min(config.training.batch_size, len(split.train))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    order = torch.randperm(len(split.train), generator=generator)
    cursor = 0
    trajectory: list[dict[str, object]] = []

    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    started = time.perf_counter()

    def record(step: int) -> None:
        train_loss, train_accuracy, _ = _evaluate(
            model, train_tokens, train_targets
        )
        test_loss, test_accuracy, correctness = _evaluate(
            model, test_tokens, test_targets
        )
        measurements = [
            measure_noisy_verifier(
                correctness,
                parameters,
                repetitions=config.verifier.repetitions,
                seed=config.verifier.seed + step * 1009 + index,
            ).to_dict()
            for index, parameters in enumerate(config.verifier.operating_points)
        ]
        row: dict[str, object] = {
            "step": step,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "verifier_measurements": measurements,
        }
        trajectory.append(row)
        print(
            f"step={step} train_accuracy={train_accuracy:.4f} "
            f"test_accuracy={test_accuracy:.4f}"
        )

    record(0)
    for step in range(1, config.training.steps + 1):
        if cursor + batch_size > len(split.train):
            order = torch.randperm(len(split.train), generator=generator)
            cursor = 0
        batch_indices = order[cursor : cursor + batch_size].to(selected_device)
        cursor += batch_size

        if config.training.warmup_steps:
            learning_rate_scale = min(step / config.training.warmup_steps, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = config.training.learning_rate * learning_rate_scale

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_tokens.index_select(0, batch_indices))
        loss = nn.functional.cross_entropy(
            logits,
            train_targets.index_select(0, batch_indices),
        )
        loss.backward()
        optimizer.step()

        if (
            step % config.training.evaluation_interval == 0
            or step == config.training.steps
        ):
            record(step)

    runtime_seconds = time.perf_counter() - started
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(selected_device))
        if selected_device.type == "cuda"
        else 0
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    final_metrics = trajectory[-1]
    best_metrics = max(
        trajectory,
        key=lambda row: (cast(float, row["test_accuracy"]), -cast(int, row["step"])),
    )
    return {
        "environment_updates": {
            "device": str(selected_device),
            "training_runtime_seconds": runtime_seconds,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "torch_deterministic": {
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            },
        },
        "dataset": {
            "name": f"modular_addition_mod_{split.modulus}",
            "operation": "modular_addition",
            "modulus": split.modulus,
            "table_size": split.modulus**2,
            "train_examples": len(split.train),
            "test_examples": len(split.test),
            **split.fingerprints(),
            "tokenization": "[a, +, b, =] -> (a + b) mod p",
            "split_method": "seeded shuffle of the complete operation table",
            "split_seed": config.seed,
            "train_fraction": config.task.train_fraction,
        },
        "model": {
            "name": "tiny_causal_transformer",
            "framework": "pytorch",
            "layers": config.model.layers,
            "d_model": config.model.d_model,
            "heads": config.model.heads,
            "d_ff": config.model.d_ff,
            "dropout": config.model.dropout,
            "sequence_length": SEQUENCE_LENGTH,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "parameter_dtype": str(next(model.parameters()).dtype),
        },
        "optimization": {
            "optimizer": "AdamW",
            "steps": config.training.steps,
            "configured_batch_size": config.training.batch_size,
            "effective_batch_size": batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "warmup_steps": config.training.warmup_steps,
            "evaluation_interval": config.training.evaluation_interval,
            "loss": "cross_entropy_on_answer_token",
        },
        "definitions": {
            "base_accuracy": "a = P(c=1)",
            "verifier_true_positive_rate": "alpha = P(V=1 | c=1)",
            "verifier_false_positive_rate": "beta = P(V=1 | c=0)",
            "accepted_accuracy": "a_prime = P(c=1 | V=1)",
            "accepted_accuracy_formula": (
                "a_prime = alpha * a / "
                "(alpha * a + beta * (1 - a))"
            ),
        },
        "trajectory": trajectory,
        "results": {
            "final_metrics": final_metrics,
            "best_metrics": best_metrics,
            "evaluation_checkpoint_count": len(trajectory),
        },
    }
