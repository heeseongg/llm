from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llmfs.data.dataset import SimpleTextDataset
from llmfs.model.transformer import TransformerLM
from llmfs.train.trainer import Trainer


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping")
    return data


def load_train_and_val_text(config: Dict[str, Any]) -> Tuple[str, Optional[str], int]:
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})

    text_path = Path(data_cfg.get("text_path", "data/raw/input.txt"))
    if not text_path.is_absolute():
        text_path = ROOT_DIR / text_path

    encoding = data_cfg.get("encoding", "utf-8")
    text = text_path.read_text(encoding=encoding)

    block_size = int(data_cfg.get("block_size", config.get("model", {}).get("max_seq_len", 128)))
    eval_every = int(train_cfg.get("eval_every", 0))
    val_text: Optional[str] = None

    val_text_path_value = data_cfg.get("val_text_path")
    if val_text_path_value is not None:
        val_text_path = Path(val_text_path_value)
        if not val_text_path.is_absolute():
            val_text_path = ROOT_DIR / val_text_path
        val_text = val_text_path.read_text(encoding=encoding)
    elif eval_every > 0:
        val_split_value = data_cfg.get("val_split")
        if val_split_value is not None:
            val_split = float(val_split_value)
            if not (0.0 < val_split < 1.0):
                raise ValueError("data.val_split must be between 0 and 1 (exclusive).")
            split_idx = int(len(text) * (1.0 - val_split))
        else:
            auto_val_len = max(block_size + 1, int(len(text) * 0.1))
            split_idx = len(text) - auto_val_len

        min_len = block_size + 1
        if split_idx < min_len or (len(text) - split_idx) < min_len:
            raise ValueError(
                "Not enough text to create train/validation split. "
                "Use a larger corpus, smaller block_size, or provide data.val_text_path."
            )

        val_text = text[split_idx:]
        text = text[:split_idx]

    return text, val_text, block_size


def build_dataloader(config: Dict[str, Any], text: str, block_size: int) -> DataLoader:
    train_cfg = config.get("train", {})
    dataset = SimpleTextDataset(text=text, block_size=block_size)

    batch_size = int(train_cfg.get("batch_size", 32))
    shuffle = bool(train_cfg.get("shuffle", True))
    num_workers = int(train_cfg.get("num_workers", 0))
    drop_last = bool(train_cfg.get("drop_last", True))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def encode_text(text: str, stoi: Dict[str, int]) -> List[int]:
    unknown = sorted({ch for ch in text if ch not in stoi})
    if unknown:
        preview = "".join(unknown[:20])
        raise ValueError(
            f"Validation text includes {len(unknown)} unknown characters not in training vocab. "
            f"Preview: {preview!r}"
        )
    return [stoi[ch] for ch in text]


def build_model(config: Dict[str, Any], dataloader: DataLoader) -> TransformerLM:
    model_cfg = config.get("model", {})
    dataset = dataloader.dataset

    vocab_size = int(model_cfg.get("vocab_size", dataset.vocab_size))
    max_seq_len = int(model_cfg.get("max_seq_len", dataset.block_size))
    d_model = int(model_cfg.get("d_model", 256))
    n_heads = int(model_cfg.get("n_heads", 8))
    n_layers = int(model_cfg.get("n_layers", 6))
    d_ff = int(model_cfg.get("d_ff", 1024))
    dropout = float(model_cfg.get("dropout", 0.1))

    return TransformerLM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    config = load_yaml(config_path)
    train_text, val_text, block_size = load_train_and_val_text(config)
    dataloader = build_dataloader(config, text=train_text, block_size=block_size)
    model = build_model(config, dataloader)

    trainer_cfg: Dict[str, Any] = dict(config.get("train", {}))
    trainer_cfg.update(config.get("trainer", {}))

    val_token_ids: Optional[List[int]] = None
    if val_text is not None:
        dataset = dataloader.dataset
        val_token_ids = encode_text(val_text, dataset.stoi)
        print(
            f"[train] validation enabled: {len(val_token_ids)} tokens "
            f"(eval_every={int(trainer_cfg.get('eval_every', 0))})"
        )

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        config=trainer_cfg,
        val_token_ids=val_token_ids,
        eval_block_size=block_size,
    )
    trainer.train()


if __name__ == "__main__":
    main()
