from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.transformer import TransformerLM


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping")
    return data


def infer_n_layers(state_dict: Dict[str, torch.Tensor]) -> int:
    layer_ids = set()
    for key in state_dict:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))
    return max(layer_ids) + 1 if layer_ids else 0


def infer_n_heads(d_model: int, requested: int | None) -> int:
    if requested is not None and requested > 0 and d_model % requested == 0:
        return requested
    for candidate in (16, 12, 10, 8, 6, 5, 4, 3, 2, 1):
        if d_model % candidate == 0:
            return candidate
    return 1


def infer_model_kwargs(
    state_dict: Dict[str, torch.Tensor],
    model_cfg: Dict[str, Any],
) -> Dict[str, int | float]:
    vocab_size = int(state_dict["lm_head.weight"].shape[0])
    d_model = int(state_dict["lm_head.weight"].shape[1])
    max_seq_len = int(state_dict["pos_emb.weight"].shape[0])
    n_layers = int(model_cfg.get("n_layers", infer_n_layers(state_dict)))

    if "blocks.0.ffn.net.0.weight" in state_dict:
        d_ff = int(state_dict["blocks.0.ffn.net.0.weight"].shape[0])
    else:
        d_ff = int(model_cfg.get("d_ff", d_model * 4))

    requested_heads = model_cfg.get("n_heads")
    n_heads = infer_n_heads(d_model, int(requested_heads) if requested_heads is not None else None)

    dropout = float(model_cfg.get("dropout", 0.0))

    return {
        "vocab_size": int(model_cfg.get("vocab_size", vocab_size)),
        "max_seq_len": int(model_cfg.get("max_seq_len", max_seq_len)),
        "d_model": int(model_cfg.get("d_model", d_model)),
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": int(model_cfg.get("d_ff", d_ff)),
        "dropout": dropout,
    }


def build_vocab_from_text(text_path: Path, encoding: str = "utf-8") -> Tuple[Dict[str, int], Dict[int, str]]:
    text = text_path.read_text(encoding=encoding)
    vocab: List[str] = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def load_vocab(
    checkpoint: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    if "stoi" in checkpoint and "itos" in checkpoint:
        stoi_raw = checkpoint["stoi"]
        itos_raw = checkpoint["itos"]
        stoi = {str(k): int(v) for k, v in stoi_raw.items()}
        itos = {int(k): str(v) for k, v in itos_raw.items()}
        return stoi, itos

    if "vocab" in checkpoint:
        vocab = [str(ch) for ch in checkpoint["vocab"]]
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = {i: ch for ch, i in stoi.items()}
        return stoi, itos

    data_cfg = config.get("data", {})
    text_path_value = data_cfg.get("text_path")
    if text_path_value is None:
        raise ValueError("Vocabulary not found in checkpoint. Provide --config with data.text_path.")

    text_path = Path(text_path_value)
    if not text_path.is_absolute():
        text_path = ROOT_DIR / text_path
    encoding = str(data_cfg.get("encoding", "utf-8"))
    return build_vocab_from_text(text_path, encoding)


def generate(
    model: TransformerLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    model.eval()
    idx = input_ids
    max_seq_len = model.max_seq_len

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_seq_len:]
            logits = model(idx_cond)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)

    return idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT_DIR / checkpoint_path

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    config: Dict[str, Any] = {}
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT_DIR / config_path
        config = load_yaml(config_path)

    model_cfg: Dict[str, Any] = {}
    if isinstance(config.get("model"), dict):
        model_cfg.update(config["model"])
    ckpt_cfg = checkpoint.get("config")
    if isinstance(ckpt_cfg, dict):
        if isinstance(ckpt_cfg.get("model"), dict):
            model_cfg.update(ckpt_cfg["model"])
        if isinstance(ckpt_cfg.get("model_config"), dict):
            model_cfg.update(ckpt_cfg["model_config"])
    if isinstance(checkpoint.get("model_config"), dict):
        model_cfg.update(checkpoint["model_config"])

    model_kwargs = infer_model_kwargs(state_dict, model_cfg)
    model = TransformerLM(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    stoi, itos = load_vocab(checkpoint, config)

    prompt = args.prompt if args.prompt is not None else input("prompt> ")
    if any(ch not in stoi for ch in prompt):
        unknown = sorted({ch for ch in prompt if ch not in stoi})
        raise ValueError(f"Prompt contains unknown characters: {unknown}")

    input_ids = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)
    output_ids = generate(model, input_ids, max_new_tokens=args.max_new_tokens)
    output_text = "".join(itos[int(i)] for i in output_ids[0].tolist())
    print(output_text)


if __name__ == "__main__":
    main()
