from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

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


def build_dataloader(config: Dict[str, Any]) -> DataLoader:
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})

    text_path = Path(data_cfg.get("text_path", "data/raw/input.txt"))
    if not text_path.is_absolute():
        text_path = ROOT_DIR / text_path

    encoding = data_cfg.get("encoding", "utf-8")
    text = text_path.read_text(encoding=encoding)

    block_size = int(data_cfg.get("block_size", config.get("model", {}).get("max_seq_len", 128)))
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
    dataloader = build_dataloader(config)
    model = build_model(config, dataloader)

    trainer_cfg: Dict[str, Any] = dict(config.get("train", {}))
    trainer_cfg.update(config.get("trainer", {}))

    trainer = Trainer(model=model, dataloader=dataloader, config=trainer_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
