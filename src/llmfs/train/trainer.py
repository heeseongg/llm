from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.config = config

        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        lr = float(config.get("lr", 3e-4))
        weight_decay = float(config.get("weight_decay", 0.0))
        betas = config.get("betas", (0.9, 0.95))
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

        self.grad_clip = float(config.get("grad_clip", 1.0))
        self.num_epochs = int(config.get("num_epochs", 1))
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.save_every = int(config.get("save_every", 0))

        self.global_step = 0

    def train(self) -> None:
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch in self.dataloader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                self.global_step += 1

                if self.save_every > 0 and self.global_step % self.save_every == 0:
                    self.save_checkpoint(epoch=epoch + 1, step=self.global_step)

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ) -> str:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if path is None:
            epoch_value = epoch if epoch is not None else 0
            step_value = step if step is not None else self.global_step
            filename = f"ckpt_epoch{epoch_value}_step{step_value}.pt"
            ckpt_path = self.checkpoint_dir / filename
        else:
            ckpt_path = Path(path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
            "epoch": epoch,
            "step": step,
        }
        torch.save(payload, ckpt_path)
        return str(ckpt_path)
