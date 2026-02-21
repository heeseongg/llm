from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from llmfs.eval.perplexity import evaluate_perplexity
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: Dict[str, Any],
        val_token_ids: Optional[Sequence[int]] = None,
        eval_block_size: Optional[int] = None,
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
        self.eval_every = int(config.get("eval_every", 0))
        self.save_best_checkpoint = bool(config.get("save_best_checkpoint", True))
        self.best_checkpoint_name = str(config.get("best_checkpoint_name", "best.pt"))
        default_eval_batch_size = int(getattr(dataloader, "batch_size", 32) or 32)
        self.eval_batch_size = int(config.get("eval_batch_size", default_eval_batch_size))
        dataset_block_size = int(getattr(dataloader.dataset, "block_size", 128))
        self.eval_block_size = int(eval_block_size if eval_block_size is not None else dataset_block_size)
        self.val_token_ids = list(val_token_ids) if val_token_ids is not None else None

        self.global_step = 0
        self.best_val_ppl: Optional[float] = None

        if self.eval_every > 0 and self.val_token_ids is None:
            raise ValueError("eval_every is set, but no validation tokens were provided.")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be > 0")
        if self.val_token_ids is not None and len(self.val_token_ids) < self.eval_block_size + 1:
            raise ValueError("Validation tokens length must be at least eval_block_size + 1.")
        if not self.best_checkpoint_name:
            raise ValueError("best_checkpoint_name must not be empty")

    def run_validation(self, epoch: int, train_loss: float) -> Optional[Tuple[float, float]]:
        if self.val_token_ids is None:
            return None

        val_ppl, val_nll = evaluate_perplexity(
            model=self.model,
            token_ids=self.val_token_ids,
            block_size=self.eval_block_size,
            batch_size=self.eval_batch_size,
            device=self.device,
        )
        print(
            f"[eval] epoch={epoch} step={self.global_step} "
            f"train_loss={train_loss:.4f} val_nll={val_nll:.4f} val_ppl={val_ppl:.4f}"
        )
        return val_ppl, val_nll

    def maybe_save_best_checkpoint(self, epoch: int, val_ppl: float) -> None:
        if not self.save_best_checkpoint:
            return

        is_best = self.best_val_ppl is None or val_ppl < self.best_val_ppl
        if not is_best:
            return

        self.best_val_ppl = val_ppl
        best_path = Path(self.best_checkpoint_name)
        if not best_path.is_absolute():
            best_path = self.checkpoint_dir / best_path

        saved_path = self.save_checkpoint(path=str(best_path), epoch=epoch, step=self.global_step)
        print(f"[best] epoch={epoch} step={self.global_step} val_ppl={val_ppl:.4f} saved={saved_path}")

    def train(self) -> None:
        self.model.train()
        last_train_loss = float("nan")
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
                last_train_loss = float(loss.item())

                if self.save_every > 0 and self.global_step % self.save_every == 0:
                    self.save_checkpoint(epoch=epoch + 1, step=self.global_step)
                if self.eval_every > 0 and self.global_step % self.eval_every == 0:
                    metrics = self.run_validation(epoch=epoch + 1, train_loss=last_train_loss)
                    if metrics is not None:
                        self.maybe_save_best_checkpoint(epoch=epoch + 1, val_ppl=metrics[0])

        # Always persist the final state, even when save_every is larger than total steps.
        self.save_checkpoint(epoch=self.num_epochs, step=self.global_step)
        if (
            self.eval_every > 0
            and self.val_token_ids is not None
            and self.global_step % self.eval_every != 0
        ):
            metrics = self.run_validation(epoch=self.num_epochs, train_loss=last_train_loss)
            if metrics is not None:
                self.maybe_save_best_checkpoint(epoch=self.num_epochs, val_ppl=metrics[0])

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
            "best_val_ppl": self.best_val_ppl,
        }
        torch.save(payload, ckpt_path)
        return str(ckpt_path)
