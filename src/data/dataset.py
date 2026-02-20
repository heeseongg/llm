from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset


class SimpleTextDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(self, text: str, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        if len(text) < block_size + 1:
            raise ValueError("text length must be at least block_size + 1")

        self.block_size = block_size
        self.vocab: List[str] = sorted(set(text))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}
        self.data: List[int] = [self.stoi[ch] for ch in text]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")

        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]

        input_ids = torch.tensor(x, dtype=torch.long)
        target_ids = torch.tensor(y, dtype=torch.long)
        return {"input_ids": input_ids, "target_ids": target_ids}
