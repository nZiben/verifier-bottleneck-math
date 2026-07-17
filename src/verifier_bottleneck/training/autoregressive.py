"""Reusable PyTorch utilities for supervised autoregressive experiments."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import torch  # type: ignore[import-not-found]
from torch import nn

AutoregressiveRow = tuple[list[int], int]
ForwardBatch = Callable[[Any, Any], Any]


def set_deterministic_seed(seed: int) -> None:
    """Seed Python and PyTorch and request deterministic kernels."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> Any:
    """Resolve ``auto`` and fail clearly when requested CUDA is absent."""
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("device=cuda was requested, but CUDA is unavailable")
    if requested == "auto":
        requested = "cuda" if cuda_available else "cpu"
    return torch.device(requested)


def collate_autoregressive(
    rows: Sequence[AutoregressiveRow], *, pad_id: int, device: Any
) -> tuple[Any, Any, Any]:
    """Pad next-token rows and mask prompt tokens from the loss."""
    maximum_length = max(len(tokens) for tokens, _ in rows) - 1
    inputs = torch.full((len(rows), maximum_length), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), maximum_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), maximum_length), dtype=torch.long)
    for row_index, (tokens, prompt_length) in enumerate(rows):
        length = len(tokens) - 1
        inputs[row_index, :length] = torch.tensor(tokens[:-1], dtype=torch.long)
        labels[row_index, prompt_length - 1 : length] = torch.tensor(
            tokens[prompt_length:], dtype=torch.long
        )
        attention_mask[row_index, :length] = 1
    return inputs.to(device), labels.to(device), attention_mask.to(device)


def autoregressive_loss(logits: Any, labels: Any, *, force_float: bool = False) -> Any:
    """Compute summed-mean next-token cross entropy with ignored prompt labels."""
    values = logits.float() if force_float else logits
    return nn.functional.cross_entropy(
        values.reshape(-1, values.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def teacher_forced_metrics(
    rows: Sequence[AutoregressiveRow],
    *,
    pad_id: int,
    device: Any,
    batch_size: int,
    forward_batch: ForwardBatch,
    force_float_loss: bool = False,
) -> tuple[float, float]:
    """Return token-level loss and accuracy for an encoded dataset."""
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            inputs, labels, attention_mask = collate_autoregressive(
                rows[start : start + batch_size], pad_id=pad_id, device=device
            )
            logits = forward_batch(inputs, attention_mask)
            mask = labels.ne(-100)
            values = logits.float() if force_float_loss else logits
            loss = nn.functional.cross_entropy(
                values.reshape(-1, values.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_correct += int(logits.argmax(dim=-1).eq(labels).logical_and(mask).sum().item())
            total_tokens += int(mask.sum().item())
    return total_loss / total_tokens, total_correct / total_tokens


def apply_linear_warmup(
    optimizer: Any, *, step: int, warmup_steps: int, learning_rate: float
) -> None:
    """Apply a linear warmup multiplier to all optimizer parameter groups."""
    if warmup_steps:
        scale = min(step / warmup_steps, 1.0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * scale
