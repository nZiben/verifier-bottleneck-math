"""Scratch training for isolated A/B skills and unseen symbolic Countdown A+B."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch  # type: ignore[import-not-found]
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]

from verifier_bottleneck.data.countdown_composition import (
    ArithmeticPuzzle,
    verify_postfix,
)
from verifier_bottleneck.evaluation.countdown import (
    PostfixState,
    run_temperature_sweep,
    sample_index_at_temperature,
)
from verifier_bottleneck.experiments.countdown_composition import (
    CompositionVocabulary,
    CountdownCompositionConfig,
    build_countdown_composition_dataset,
)
from verifier_bottleneck.experiments.countdown_protocol import replay_example_count
from verifier_bottleneck.live_logging import live_log
from verifier_bottleneck.training.autoregressive import (
    apply_linear_warmup,
    autoregressive_loss,
    collate_autoregressive,
    teacher_forced_metrics,
)
from verifier_bottleneck.training.autoregressive import (
    select_device as _select_device,
)
from verifier_bottleneck.training.autoregressive import (
    set_deterministic_seed as _set_seed,
)
from verifier_bottleneck.training.countdown_payload import build_countdown_training_payload


class NanoArithmeticTransformer(nn.Module):
    """Small decoder-style transformer with tied token embeddings."""

    def __init__(self, config: CountdownCompositionConfig, vocabulary_size: int) -> None:
        super().__init__()
        model = config.model
        self.token_embedding = nn.Embedding(vocabulary_size, model.d_model)
        self.position_embedding = nn.Embedding(model.block_size, model.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=model.d_model,
            nhead=model.heads,
            dim_feedforward=model.d_ff,
            dropout=model.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=model.layers)
        self.final_norm = nn.LayerNorm(model.d_model)

    def forward(self, tokens: Any, *, padding_mask: Any | None = None) -> Any:
        """Return next-token logits at each sequence position."""
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return nn.functional.linear(self.final_norm(hidden), self.token_embedding.weight)


def _collate(
    rows: Sequence[tuple[list[int], int]], *, pad_id: int, device: Any
) -> tuple[Any, Any, Any]:
    inputs, labels, attention_mask = collate_autoregressive(rows, pad_id=pad_id, device=device)
    return inputs, labels, attention_mask.eq(0)


def _teacher_forced_metrics(
    model: NanoArithmeticTransformer,
    rows: Sequence[tuple[list[int], int]],
    *,
    pad_id: int,
    device: Any,
    batch_size: int = 512,
) -> tuple[float, float]:
    model.eval()
    return teacher_forced_metrics(
        rows,
        pad_id=pad_id,
        device=device,
        batch_size=batch_size,
        forward_batch=lambda inputs, attention_mask: model(
            inputs, padding_mask=attention_mask.eq(0)
        ),
    )


def _generate_batch(
    model: NanoArithmeticTransformer,
    puzzles: Sequence[ArithmeticPuzzle],
    *,
    vocabulary: CompositionVocabulary,
    value_to_symbol: Mapping[int, str] | None,
    temperature: float,
    seed: int,
    device: Any,
) -> list[bool]:
    mapping = vocabulary.token_to_id
    prompts = []
    for puzzle in puzzles:
        encoded, prompt_length = vocabulary.encode_countdown(
            puzzle, value_to_symbol=value_to_symbol
        )
        prompts.append(encoded[:prompt_length])
    sequences = torch.tensor(prompts, dtype=torch.long, device=device)
    states = [PostfixState() for _ in puzzles]
    random_sources = [
        random.Random(seed + row_index * 1_000_000_007) for row_index in range(len(puzzles))
    ]
    maximum_tokens = 2 * max(len(puzzle.numbers) for puzzle in puzzles)
    pad_id = mapping["<PAD>"]
    model.eval()
    with torch.no_grad():
        for _ in range(maximum_tokens):
            logits = model(sequences)[:, -1, :]
            next_ids = [pad_id] * len(puzzles)
            for row_index, puzzle in enumerate(puzzles):
                state = states[row_index]
                if state.finished:
                    continue
                legal_actions = state.legal_actions(len(puzzle.numbers))
                legal = [mapping[action] for action in legal_actions]
                legal_logits = logits[row_index, legal].float().tolist()
                random_source = random_sources[row_index]
                random_value = random_source.random() if temperature > 0.0 else 0.0
                chosen = legal[
                    sample_index_at_temperature(
                        legal_logits,
                        temperature=temperature,
                        random_value=random_value,
                    )
                ]
                next_ids[row_index] = chosen
                state.apply(vocabulary.tokens[chosen])
            sequences = torch.cat(
                (
                    sequences,
                    torch.tensor(next_ids, dtype=torch.long, device=device).unsqueeze(1),
                ),
                dim=1,
            )
            if all(state.finished for state in states):
                break
    return [
        state.finished and verify_postfix(puzzle, tuple(state.tokens))
        for puzzle, state in zip(puzzles, states, strict=True)
    ]


def _decode_exact_accuracy(
    model: NanoArithmeticTransformer,
    *,
    decode_rows: Sequence[tuple[list[int], int]],
    expected_values: Sequence[int],
    vocabulary: CompositionVocabulary,
    device: Any,
) -> tuple[int, int]:
    mapping = vocabulary.token_to_id
    prompt_ids = [tokens[:prompt_length] for tokens, prompt_length in decode_rows]
    inputs = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    candidate_ids = [mapping[f"N{value}"] for value in expected_values]
    model.eval()
    with torch.no_grad():
        logits = model(inputs)[:, -1, :]
        predictions = logits[:, candidate_ids].argmax(dim=-1).tolist()
    expected_by_row = [candidate_ids.index(mapping[f"N{value}"]) for value in expected_values]
    solved = sum(
        int(prediction == expected)
        for prediction, expected in zip(predictions, expected_by_row, strict=True)
    )
    return solved, len(expected_values)


def _temperature_sweep(
    model: NanoArithmeticTransformer,
    puzzles: Sequence[ArithmeticPuzzle],
    *,
    config: CountdownCompositionConfig,
    vocabulary: CompositionVocabulary,
    value_to_symbol: Mapping[int, str],
    device: Any,
    tensorboard_writer: Any,
    proposal_outcomes_path: Path,
) -> list[dict[str, object]]:
    return run_temperature_sweep(
        puzzles,
        settings=config.evaluation,
        value_to_symbol=value_to_symbol,
        generate_batch=lambda batch, symbols, temperature, seed: _generate_batch(
            model,
            batch,
            vocabulary=vocabulary,
            value_to_symbol=symbols,
            temperature=temperature,
            seed=seed,
            device=device,
        ),
        writer=tensorboard_writer,
        proposal_outcomes_path=proposal_outcomes_path,
    )


def train_countdown_composition_transformer(
    config: CountdownCompositionConfig,
    *,
    device: str,
    tensorboard_directory: Path,
    checkpoint_path: Path,
    proposal_outcomes_path: Path,
    codebook_path: Path,
) -> dict[str, object]:
    """Train only A and B, select without AB, then evaluate unseen A+B."""
    config.validate()
    _set_seed(config.seed)
    selected_device = _select_device(device)
    split = build_countdown_composition_dataset(config)
    vocabulary = CompositionVocabulary.build()
    mapping = vocabulary.token_to_id
    decode_rows = [vocabulary.encode_decode(example) for example in split.decode_examples]
    numeric_train_rows = [
        vocabulary.encode_countdown(example, value_to_symbol=None)
        for example in split.numeric_train
    ]
    numeric_validation_rows = [
        vocabulary.encode_countdown(example, value_to_symbol=None)
        for example in split.numeric_validation
    ]
    numeric_test_rows = [
        vocabulary.encode_countdown(example, value_to_symbol=None) for example in split.numeric_test
    ]
    symbolic_test_rows = [
        vocabulary.encode_countdown(example, value_to_symbol=split.value_to_symbol)
        for example in split.symbolic_test
    ]
    codebook_path.parent.mkdir(parents=True, exist_ok=True)
    codebook_path.write_text(
        json.dumps(split.symbol_to_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model = NanoArithmeticTransformer(cast(Any, config), len(vocabulary.tokens)).to(selected_device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    def new_optimizer() -> Any:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    optimizer = new_optimizer()
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(config.seed)
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_directory), flush_secs=10)
    tensorboard_writer.add_text(
        "run/config",
        f"```json\n{json.dumps(config.to_dict(), indent=2, sort_keys=True)}\n```",
        0,
    )
    tensorboard_writer.add_text("run/device", str(selected_device), 0)
    tensorboard_writer.add_scalar("model/parameter_count", parameter_count, 0)
    tensorboard_writer.flush()
    trajectory: list[dict[str, object]] = []
    best_numeric_b_validation_loss = math.inf
    best_checkpoint_step = 0
    checkpoints_without_improvement = 0
    stopped_early = False
    started = time.perf_counter()
    last_log_at = started
    last_log_step = 0
    expected_values = [example.value for example in split.decode_examples]
    numeric_train_evaluation_rows = numeric_train_rows[: min(512, len(numeric_train_rows))]

    def record_skill_a(step: int) -> int:
        decode_loss, decode_token_accuracy = _teacher_forced_metrics(
            model,
            decode_rows,
            pad_id=mapping["<PAD>"],
            device=selected_device,
        )
        decode_solved, decode_total = _decode_exact_accuracy(
            model,
            decode_rows=decode_rows,
            expected_values=expected_values,
            vocabulary=vocabulary,
            device=selected_device,
        )
        row: dict[str, object] = {
            "step": step,
            "phase": "skill_a_pretraining",
            "decode_a_token_loss": decode_loss,
            "decode_a_token_accuracy": decode_token_accuracy,
            "decode_a_exact_accuracy": decode_solved / decode_total,
            "is_best_checkpoint": False,
        }
        trajectory.append(row)
        for name, value in row.items():
            if name != "step" and isinstance(value, int | float | bool):
                tensorboard_writer.add_scalar(f"skill_a_pretraining/{name}", value, step)
        tensorboard_writer.flush()
        live_log(
            f"composition phase=A step={step} decode_A={decode_solved}/{decode_total} "
            f"decode_A_loss={decode_loss:.4f}"
        )
        return decode_solved

    def record_skill_b(step: int) -> bool:
        nonlocal best_checkpoint_step, best_numeric_b_validation_loss
        decode_loss, decode_token_accuracy = _teacher_forced_metrics(
            model,
            decode_rows,
            pad_id=mapping["<PAD>"],
            device=selected_device,
        )
        numeric_loss, numeric_accuracy = _teacher_forced_metrics(
            model,
            numeric_validation_rows,
            pad_id=mapping["<PAD>"],
            device=selected_device,
        )
        numeric_train_loss, numeric_train_accuracy = _teacher_forced_metrics(
            model,
            numeric_train_evaluation_rows,
            pad_id=mapping["<PAD>"],
            device=selected_device,
        )
        combined_loss = (decode_loss + numeric_loss) / 2.0
        combined_accuracy = (decode_token_accuracy + numeric_accuracy) / 2.0
        combined_train_loss = (decode_loss + numeric_train_loss) / 2.0
        combined_train_accuracy = (decode_token_accuracy + numeric_train_accuracy) / 2.0
        decode_solved, decode_total = _decode_exact_accuracy(
            model,
            decode_rows=decode_rows,
            expected_values=expected_values,
            vocabulary=vocabulary,
            device=selected_device,
        )
        skill_a_retained = decode_solved == decode_total
        improved = (
            skill_a_retained
            and numeric_loss
            < best_numeric_b_validation_loss - config.training.early_stopping_min_delta
        )
        if improved:
            best_numeric_b_validation_loss = numeric_loss
            best_checkpoint_step = step
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "validation_token_loss": combined_loss,
                    "validation_token_accuracy": combined_accuracy,
                    "numeric_b_validation_loss": numeric_loss,
                    "decode_a_exact_accuracy": decode_solved / decode_total,
                },
                checkpoint_path,
            )
        row: dict[str, object] = {
            "step": step,
            "phase": "skill_b_with_a_replay",
            "train_token_loss": combined_train_loss,
            "train_token_accuracy": combined_train_accuracy,
            "validation_token_loss": combined_loss,
            "validation_token_accuracy": combined_accuracy,
            "decode_a_token_loss": decode_loss,
            "decode_a_token_accuracy": decode_token_accuracy,
            "decode_a_exact_accuracy": decode_solved / decode_total,
            "numeric_b_train_loss": numeric_train_loss,
            "numeric_b_train_accuracy": numeric_train_accuracy,
            "numeric_b_validation_loss": numeric_loss,
            "numeric_b_validation_accuracy": numeric_accuracy,
            "is_best_checkpoint": improved,
        }
        trajectory.append(row)
        for name, value in row.items():
            if name != "step" and isinstance(value, int | float | bool):
                tensorboard_writer.add_scalar(f"checkpoint/{name}", value, step)
        tensorboard_writer.flush()
        live_log(
            f"composition phase=B step={step} decode_A={decode_solved}/{decode_total} "
            f"numeric_B_validation_loss={numeric_loss:.4f} "
            f"A_retained={skill_a_retained} best={improved}"
        )
        return improved

    live_log(
        f"composition run started device={selected_device} parameters={parameter_count} "
        f"decode_A={len(decode_rows)} numeric_B_train={len(numeric_train_rows)}"
    )
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    decode_solved = record_skill_a(0)
    skill_a_pretraining_steps = 0
    while decode_solved < len(decode_rows):
        skill_a_pretraining_steps += 1
        decode_indices = torch.randint(
            len(decode_rows),
            (config.training.batch_size,),
            generator=batch_generator,
        ).tolist()
        selected_rows = [decode_rows[index] for index in decode_indices]
        inputs, labels, padding_mask = _collate(
            selected_rows, pad_id=mapping["<PAD>"], device=selected_device
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, padding_mask=padding_mask)
        loss = autoregressive_loss(logits, labels)
        loss.backward()
        optimizer.step()
        if (
            skill_a_pretraining_steps % config.training.skill_a_pretraining_evaluation_interval == 0
            or skill_a_pretraining_steps == config.training.skill_a_pretraining_max_steps
        ):
            decode_solved = record_skill_a(skill_a_pretraining_steps)
        if skill_a_pretraining_steps >= config.training.skill_a_pretraining_max_steps:
            break
    if decode_solved < len(decode_rows):
        raise RuntimeError(
            "Skill A pretraining did not reach exact 14/14 accuracy within "
            f"{config.training.skill_a_pretraining_max_steps} steps"
        )
    live_log(
        f"composition phase=A complete steps={skill_a_pretraining_steps} "
        f"decode_A={decode_solved}/{len(decode_rows)}"
    )

    optimizer = new_optimizer()
    replay_count = replay_example_count(
        config.training.batch_size, config.training.skill_a_replay_fraction
    )
    numeric_count = config.training.batch_size - replay_count
    record_skill_b(skill_a_pretraining_steps)
    last_log_at = time.perf_counter()
    last_log_step = 0
    for phase_step in range(1, config.training.steps + 1):
        global_step = skill_a_pretraining_steps + phase_step
        decode_indices = torch.randint(
            len(decode_rows), (replay_count,), generator=batch_generator
        ).tolist()
        numeric_indices = torch.randint(
            len(numeric_train_rows), (numeric_count,), generator=batch_generator
        ).tolist()
        selected_rows = [decode_rows[index] for index in decode_indices]
        selected_rows.extend(numeric_train_rows[index] for index in numeric_indices)
        inputs, labels, padding_mask = _collate(
            selected_rows, pad_id=mapping["<PAD>"], device=selected_device
        )
        apply_linear_warmup(
            optimizer,
            step=phase_step,
            warmup_steps=config.training.warmup_steps,
            learning_rate=config.training.learning_rate,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, padding_mask=padding_mask)
        loss = autoregressive_loss(logits, labels)
        loss.backward()
        optimizer.step()
        if phase_step == 1 or phase_step % config.training.log_interval == 0:
            logged_at = time.perf_counter()
            live_log(
                f"composition phase=B step={phase_step}/{config.training.steps} "
                f"global_step={global_step} A_replay={replay_count} B={numeric_count} "
                f"batch_loss={float(loss.item()):.4f} "
                f"steps_per_second="
                f"{(phase_step - last_log_step) / max(logged_at - last_log_at, 1e-9):.2f}"
            )
            tensorboard_writer.add_scalar(
                "skill_b_phase/batch_loss", float(loss.item()), global_step
            )
            tensorboard_writer.add_scalar(
                "skill_b_phase/learning_rate",
                float(optimizer.param_groups[0]["lr"]),
                global_step,
            )
            tensorboard_writer.flush()
            last_log_at = logged_at
            last_log_step = phase_step
        if (
            phase_step % config.training.evaluation_interval == 0
            or phase_step == config.training.steps
        ):
            if record_skill_b(global_step):
                checkpoints_without_improvement = 0
            else:
                checkpoints_without_improvement += 1
            if checkpoints_without_improvement >= config.training.early_stopping_patience:
                stopped_early = True
                live_log(
                    f"composition phase=B early stopping phase_step={phase_step} "
                    f"global_step={global_step} best_step={best_checkpoint_step}"
                )
                break

    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    decode_solved, decode_total = _decode_exact_accuracy(
        model,
        decode_rows=decode_rows,
        expected_values=expected_values,
        vocabulary=vocabulary,
        device=selected_device,
    )
    numeric_test_loss, numeric_test_accuracy = _teacher_forced_metrics(
        model,
        numeric_test_rows,
        pad_id=mapping["<PAD>"],
        device=selected_device,
    )
    symbolic_test_loss, symbolic_test_accuracy = _teacher_forced_metrics(
        model,
        symbolic_test_rows,
        pad_id=mapping["<PAD>"],
        device=selected_device,
    )
    measurements = _temperature_sweep(
        model,
        split.numeric_test,
        config=config,
        vocabulary=vocabulary,
        value_to_symbol=split.value_to_symbol,
        device=selected_device,
        tensorboard_writer=tensorboard_writer,
        proposal_outcomes_path=proposal_outcomes_path,
    )
    runtime_seconds = time.perf_counter() - started
    tensorboard_writer.add_scalar("run/runtime_seconds", runtime_seconds, 0)
    tensorboard_writer.flush()
    tensorboard_writer.close()
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(selected_device))
        if selected_device.type == "cuda"
        else 0
    )
    selected_metrics: dict[str, object] = {
        "step": best_checkpoint_step,
        "validation_token_loss": cast(float, checkpoint["validation_token_loss"]),
        "validation_token_accuracy": cast(float, checkpoint["validation_token_accuracy"]),
        "decode_a_exact_solved": decode_solved,
        "decode_a_exact_total": decode_total,
        "decode_a_exact_accuracy": decode_solved / decode_total,
        "numeric_b_test_token_loss": numeric_test_loss,
        "numeric_b_test_token_accuracy": numeric_test_accuracy,
        "symbolic_ab_test_token_loss": symbolic_test_loss,
        "symbolic_ab_test_token_accuracy": symbolic_test_accuracy,
    }
    return build_countdown_training_payload(
        split=split,
        environment_updates={
            "device": str(selected_device),
            "training_and_evaluation_runtime_seconds": runtime_seconds,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "torch_deterministic_algorithms": True,
        },
        model={
            "name": "nano_countdown_composition_transformer",
            "framework": "pytorch",
            "layers": config.model.layers,
            "d_model": config.model.d_model,
            "heads": config.model.heads,
            "d_ff": config.model.d_ff,
            "block_size": config.model.block_size,
            "dropout": config.model.dropout,
            "vocabulary_size": len(vocabulary.tokens),
            "parameter_count": parameter_count,
            "trainable_parameter_count": parameter_count,
            "parameter_dtype": str(next(model.parameters()).dtype),
            "pretrained": False,
        },
        optimization={
            "optimizer": "AdamW",
            "steps": config.training.steps,
            "skill_a_pretraining_max_steps": (config.training.skill_a_pretraining_max_steps),
            "skill_a_pretraining_actual_steps": skill_a_pretraining_steps,
            "skill_b_phase_steps": config.training.steps,
            "actual_steps": trajectory[-1]["step"],
            "actual_skill_b_phase_steps": (
                int(cast(int, trajectory[-1]["step"])) - skill_a_pretraining_steps
            ),
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "warmup_steps": config.training.warmup_steps,
            "early_stopping_patience": config.training.early_stopping_patience,
            "early_stopping_min_delta": config.training.early_stopping_min_delta,
            "stopped_early": stopped_early,
            "checkpoint_selection": (
                "minimum numeric-B validation loss among checkpoints retaining exact "
                "14/14 Skill-A accuracy; AB excluded"
            ),
            "task_sampling": (
                "Phase 1 uses 100% isolated A until 14/14 exact accuracy; Phase 2 uses "
                f"{replay_count}/{config.training.batch_size} isolated-A replay and "
                f"{numeric_count}/{config.training.batch_size} numeric-B examples"
            ),
            "skill_a_replay_fraction_requested": (config.training.skill_a_replay_fraction),
        },
        trajectory=trajectory,
        selected_metrics=selected_metrics,
        measurements=measurements,
        stopped_early=stopped_early,
    )
