"""Qwen LoRA training for separate A/B and unseen symbolic Countdown A+B."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch  # type: ignore[import-not-found]
from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
from transformers import (  # type: ignore[import-not-found]
    AutoModelForCausalLM,
    AutoTokenizer,
)

from verifier_bottleneck.data.countdown_composition import ArithmeticPuzzle
from verifier_bottleneck.evaluation.countdown import run_temperature_sweep
from verifier_bottleneck.experiments.countdown_protocol import distribute_replay_examples
from verifier_bottleneck.experiments.qwen_countdown_composition import (
    QwenCountdownCompositionConfig,
    build_qwen_countdown_composition_dataset,
    numeric_answer_token_sequences,
)
from verifier_bottleneck.live_logging import live_log
from verifier_bottleneck.training.autoregressive import (
    apply_linear_warmup,
    autoregressive_loss,
    collate_autoregressive,
    select_device,
    set_deterministic_seed,
    teacher_forced_metrics,
)
from verifier_bottleneck.training.countdown_payload import build_countdown_training_payload
from verifier_bottleneck.training.qwen_countdown_decoding import (
    decode_exact_accuracy,
    generate_batch,
)
from verifier_bottleneck.training.qwen_countdown_io import (
    action_token_ids,
    encoded_countdown_rows,
    encoded_decode_rows,
)


def _teacher_forced_metrics(
    model: Any,
    rows: Sequence[tuple[list[int], int]],
    *,
    pad_id: int,
    device: Any,
    batch_size: int = 32,
) -> tuple[float, float]:
    model.eval()
    return teacher_forced_metrics(
        rows,
        pad_id=pad_id,
        device=device,
        batch_size=batch_size,
        forward_batch=lambda inputs, attention_mask: model(
            input_ids=inputs,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits,
        force_float_loss=True,
    )


def _trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _temperature_sweep(
    model: Any,
    puzzles: Sequence[ArithmeticPuzzle],
    *,
    config: QwenCountdownCompositionConfig,
    tokenizer: Any,
    action_ids: Mapping[str, int],
    value_to_symbol: Mapping[int, str],
    device: Any,
    writer: Any,
    proposal_outcomes_path: Path,
) -> list[dict[str, object]]:
    return run_temperature_sweep(
        puzzles,
        settings=config.evaluation,
        value_to_symbol=value_to_symbol,
        generate_batch=lambda batch, symbols, temperature, seed: generate_batch(
            model,
            batch,
            tokenizer=tokenizer,
            action_ids=action_ids,
            value_to_symbol=symbols,
            temperature=temperature,
            seed=seed,
            device=device,
        ),
        writer=writer,
        proposal_outcomes_path=proposal_outcomes_path,
        log_prefix="Qwen composition",
    )


def train_qwen_countdown_composition(
    config: QwenCountdownCompositionConfig,
    *,
    device: str,
    tensorboard_directory: Path,
    checkpoint_path: Path,
    proposal_outcomes_path: Path,
    codebook_path: Path,
    token_map_path: Path,
) -> dict[str, object]:
    """Fine-tune only isolated A/B, then evaluate the held-out A+B composition."""
    config.validate()
    set_deterministic_seed(config.seed)
    selected_device = select_device(device)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.pretrained_model,
        revision=config.model.revision,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Qwen tokenizer does not define eos_token_id")
    tokenizer.pad_token = tokenizer.eos_token
    split = build_qwen_countdown_composition_dataset(config)
    action_ids = action_token_ids(tokenizer)
    numeric_sequences = numeric_answer_token_sequences(
        tokenizer, [example.value for example in split.decode_examples]
    )
    decode_rows = encoded_decode_rows(split.decode_examples, tokenizer, numeric_sequences)
    numeric_train_rows = encoded_countdown_rows(
        split.numeric_train,
        tokenizer=tokenizer,
        action_ids=action_ids,
        value_to_symbol=None,
    )
    numeric_validation_rows = encoded_countdown_rows(
        split.numeric_validation,
        tokenizer=tokenizer,
        action_ids=action_ids,
        value_to_symbol=None,
    )
    numeric_test_rows = encoded_countdown_rows(
        split.numeric_test,
        tokenizer=tokenizer,
        action_ids=action_ids,
        value_to_symbol=None,
    )
    symbolic_test_rows = encoded_countdown_rows(
        split.symbolic_test,
        tokenizer=tokenizer,
        action_ids=action_ids,
        value_to_symbol=split.value_to_symbol,
    )
    load_dtype = torch.float16 if selected_device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model.pretrained_model,
        revision=config.model.revision,
        trust_remote_code=False,
        torch_dtype=load_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(selected_device)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=list(config.model.target_modules),
            bias="none",
        ),
    )
    codebook_path.parent.mkdir(parents=True, exist_ok=True)
    codebook_path.write_text(
        json.dumps(split.symbol_to_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    token_map_path.write_text(
        json.dumps(
            {
                "actions": action_ids,
                "decoded_number_token_sequences": numeric_sequences,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    def new_optimizer() -> Any:
        return torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    optimizer = new_optimizer()
    scaler = torch.amp.GradScaler("cuda", enabled=selected_device.type == "cuda")
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(config.seed)
    writer = SummaryWriter(log_dir=str(tensorboard_directory), flush_secs=10)
    writer.add_text(
        "run/config",
        f"```json\n{json.dumps(config.to_dict(), indent=2, sort_keys=True)}\n```",
        0,
    )
    writer.add_text("run/device", str(selected_device), 0)
    writer.add_scalar("model/total_parameter_count", total_parameters, 0)
    writer.add_scalar("model/trainable_parameter_count", trainable_parameters, 0)
    writer.flush()
    trajectory: list[dict[str, object]] = []
    best_numeric_b_validation_loss = math.inf
    best_checkpoint_step = 0
    checkpoints_without_improvement = 0
    stopped_early = False
    pad_id = int(tokenizer.pad_token_id)
    numeric_train_evaluation_rows = numeric_train_rows[: min(256, len(numeric_train_rows))]

    def record_skill_a(step: int) -> int:
        decode_loss, decode_token_accuracy = _teacher_forced_metrics(
            model, decode_rows, pad_id=pad_id, device=selected_device
        )
        decode_solved, decode_total = decode_exact_accuracy(
            model,
            decode_examples=split.decode_examples,
            tokenizer=tokenizer,
            numeric_sequences=numeric_sequences,
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
                writer.add_scalar(f"skill_a_pretraining/{name}", value, step)
        writer.flush()
        live_log(
            f"Qwen composition phase=A step={step} "
            f"decode_A={decode_solved}/{decode_total} decode_A_loss={decode_loss:.4f}"
        )
        return decode_solved

    def record_skill_b(step: int) -> bool:
        nonlocal best_checkpoint_step, best_numeric_b_validation_loss
        decode_loss, decode_token_accuracy = _teacher_forced_metrics(
            model, decode_rows, pad_id=pad_id, device=selected_device
        )
        numeric_loss, numeric_accuracy = _teacher_forced_metrics(
            model,
            numeric_validation_rows,
            pad_id=pad_id,
            device=selected_device,
        )
        numeric_train_loss, numeric_train_accuracy = _teacher_forced_metrics(
            model,
            numeric_train_evaluation_rows,
            pad_id=pad_id,
            device=selected_device,
        )
        decode_solved, decode_total = decode_exact_accuracy(
            model,
            decode_examples=split.decode_examples,
            tokenizer=tokenizer,
            numeric_sequences=numeric_sequences,
            device=selected_device,
        )
        combined_loss = (decode_loss + numeric_loss) / 2.0
        combined_accuracy = (decode_token_accuracy + numeric_accuracy) / 2.0
        combined_train_loss = (decode_loss + numeric_train_loss) / 2.0
        combined_train_accuracy = (decode_token_accuracy + numeric_train_accuracy) / 2.0
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
                    "adapter_state_dict": _trainable_state(model),
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
                writer.add_scalar(f"checkpoint/{name}", value, step)
        writer.flush()
        live_log(
            f"Qwen composition phase=B step={step} "
            f"decode_A={decode_solved}/{decode_total} numeric_B_loss={numeric_loss:.4f} "
            f"A_retained={skill_a_retained} best={improved}"
        )
        return improved

    live_log(
        f"Qwen composition initialized total_parameters={total_parameters} "
        f"trainable_parameters={trainable_parameters}"
    )
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    decode_solved = record_skill_a(0)
    skill_a_pretraining_steps = 0
    while decode_solved < len(decode_rows):
        skill_a_pretraining_steps += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config.training.gradient_accumulation_steps):
            decode_indices = torch.randint(
                len(decode_rows),
                (config.training.micro_batch_size,),
                generator=batch_generator,
            ).tolist()
            rows = [decode_rows[index] for index in decode_indices]
            inputs, labels, attention_mask = collate_autoregressive(
                rows, pad_id=pad_id, device=selected_device
            )
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.float16,
                enabled=selected_device.type == "cuda",
            ):
                logits = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                loss = autoregressive_loss(logits, labels)
                scaled_loss = loss / config.training.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.item())
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            config.training.maximum_gradient_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        writer.add_scalar(
            "skill_a_pretraining/batch_loss",
            accumulated_loss / config.training.gradient_accumulation_steps,
            skill_a_pretraining_steps,
        )
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
        f"Qwen composition phase=A complete steps={skill_a_pretraining_steps} "
        f"decode_A={decode_solved}/{len(decode_rows)}"
    )

    optimizer = new_optimizer()
    effective_batch_size = (
        config.training.micro_batch_size * config.training.gradient_accumulation_steps
    )
    replay_counts = distribute_replay_examples(
        micro_batch_size=config.training.micro_batch_size,
        accumulation_steps=config.training.gradient_accumulation_steps,
        replay_fraction=config.training.skill_a_replay_fraction,
    )
    replay_examples_per_step = sum(replay_counts)
    numeric_examples_per_step = effective_batch_size - replay_examples_per_step
    record_skill_b(skill_a_pretraining_steps)
    last_log_at = time.perf_counter()
    last_log_step = 0
    for phase_step in range(1, config.training.steps + 1):
        global_step = skill_a_pretraining_steps + phase_step
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for decode_count in replay_counts:
            numeric_count = config.training.micro_batch_size - decode_count
            decode_indices = torch.randint(
                len(decode_rows), (decode_count,), generator=batch_generator
            ).tolist()
            numeric_indices = torch.randint(
                len(numeric_train_rows), (numeric_count,), generator=batch_generator
            ).tolist()
            rows = [decode_rows[index] for index in decode_indices]
            rows.extend(numeric_train_rows[index] for index in numeric_indices)
            inputs, labels, attention_mask = collate_autoregressive(
                rows, pad_id=pad_id, device=selected_device
            )
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.float16,
                enabled=selected_device.type == "cuda",
            ):
                logits = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                loss = autoregressive_loss(logits, labels)
                scaled_loss = loss / config.training.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.item())
        apply_linear_warmup(
            optimizer,
            step=phase_step,
            warmup_steps=config.training.warmup_steps,
            learning_rate=config.training.learning_rate,
        )
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            config.training.maximum_gradient_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        if phase_step == 1 or phase_step % config.training.log_interval == 0:
            logged_at = time.perf_counter()
            average_loss = accumulated_loss / config.training.gradient_accumulation_steps
            live_log(
                f"Qwen composition phase=B step={phase_step}/{config.training.steps} "
                f"global_step={global_step} A_replay={replay_examples_per_step} "
                f"B={numeric_examples_per_step} "
                f"batch_loss={average_loss:.4f} steps_per_second="
                f"{(phase_step - last_log_step) / max(logged_at - last_log_at, 1e-9):.3f}"
            )
            writer.add_scalar("skill_b_phase/batch_loss", average_loss, global_step)
            writer.add_scalar(
                "skill_b_phase/learning_rate",
                float(optimizer.param_groups[0]["lr"]),
                global_step,
            )
            writer.flush()
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
                    f"Qwen composition phase=B early stopping phase_step={phase_step} "
                    f"global_step={global_step} best_step={best_checkpoint_step}"
                )
                break

    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    model.load_state_dict(cast(dict[str, Any], checkpoint["adapter_state_dict"]), strict=False)
    decode_solved, decode_total = decode_exact_accuracy(
        model,
        decode_examples=split.decode_examples,
        tokenizer=tokenizer,
        numeric_sequences=numeric_sequences,
        device=selected_device,
    )
    numeric_test_loss, numeric_test_accuracy = _teacher_forced_metrics(
        model, numeric_test_rows, pad_id=pad_id, device=selected_device
    )
    symbolic_test_loss, symbolic_test_accuracy = _teacher_forced_metrics(
        model, symbolic_test_rows, pad_id=pad_id, device=selected_device
    )
    measurements = _temperature_sweep(
        model,
        split.numeric_test,
        config=config,
        tokenizer=tokenizer,
        action_ids=action_ids,
        value_to_symbol=split.value_to_symbol,
        device=selected_device,
        writer=writer,
        proposal_outcomes_path=proposal_outcomes_path,
    )
    runtime_seconds = time.perf_counter() - started
    writer.add_scalar("run/runtime_seconds", runtime_seconds, 0)
    writer.flush()
    writer.close()
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
            "name": config.model.pretrained_model,
            "revision_requested": config.model.revision,
            "revision_resolved": getattr(model.config, "_commit_hash", None),
            "framework": "transformers+peft",
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "parameter_dtype": str(next(model.parameters()).dtype),
            "pretrained": True,
            "fine_tuning": "LoRA",
            "lora_rank": config.model.lora_rank,
            "lora_alpha": config.model.lora_alpha,
            "target_modules": list(config.model.target_modules),
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
            "micro_batch_size": config.training.micro_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "effective_batch_size": (
                config.training.micro_batch_size * config.training.gradient_accumulation_steps
            ),
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "warmup_steps": config.training.warmup_steps,
            "stopped_early": stopped_early,
            "checkpoint_selection": (
                "minimum numeric-B validation loss among checkpoints retaining exact "
                "14/14 Skill-A accuracy; AB excluded"
            ),
            "task_sampling": (
                "Phase 1 uses 100% isolated A until 14/14 exact accuracy; Phase 2 uses "
                f"{replay_examples_per_step}/{effective_batch_size} isolated-A replay "
                f"and {numeric_examples_per_step}/{effective_batch_size} numeric-B examples"
            ),
            "skill_a_replay_fraction_requested": (config.training.skill_a_replay_fraction),
        },
        trajectory=trajectory,
        selected_metrics=selected_metrics,
        measurements=measurements,
        stopped_early=stopped_early,
    )
