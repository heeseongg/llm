from __future__ import annotations

import math
from typing import Iterator, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _iter_eval_batches(
    token_ids: Sequence[int],
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    num_samples = len(token_ids) - block_size
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        x_rows = [token_ids[i : i + block_size] for i in range(start, end)]
        y_rows = [token_ids[i + 1 : i + block_size + 1] for i in range(start, end)]

        input_ids = torch.tensor(x_rows, dtype=torch.long, device=device)
        target_ids = torch.tensor(y_rows, dtype=torch.long, device=device)
        yield input_ids, target_ids


def evaluate_perplexity(
    model: nn.Module,
    token_ids: Sequence[int],
    block_size: int,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> Tuple[float, float]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if len(token_ids) < block_size + 1:
        raise ValueError("token_ids length must be at least block_size + 1")

    eval_device = device
    if eval_device is None:
        try:
            eval_device = next(model.parameters()).device
        except StopIteration:
            eval_device = torch.device("cpu")

    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, target_ids in _iter_eval_batches(
            token_ids=token_ids,
            block_size=block_size,
            batch_size=batch_size,
            device=eval_device,
        ):
            logits = model(input_ids)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_tokens += int(target_ids.numel())

    if was_training:
        model.train()

    if total_tokens == 0:
        raise ValueError("No tokens were evaluated.")

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity, avg_nll
