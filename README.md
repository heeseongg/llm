# llm-from-scratch

PyTorch로 작성한 최소 Character-level GPT(Decoder-only Transformer) 학습/샘플링 프로젝트입니다.

## 현재 구현 내용

- `src/model/transformer.py`
  - `MultiHeadAttention` (causal mask)
  - `FeedForward`
  - `LayerNorm`
  - `TransformerBlock`
  - `TransformerLM` (`forward`는 logits 반환)
- `src/data/dataset.py`
  - `SimpleTextDataset` (`torch.utils.data.Dataset` 상속)
  - next-token prediction용 `input_ids`, `target_ids` 반환
  - char vocab 자동 생성 (`stoi`/`itos`)
- `src/train/trainer.py`
  - `Trainer(model, dataloader, config)`
  - `AdamW` optimizer 설정
  - 학습 루프 + cross entropy
  - gradient clipping
  - checkpoint 저장 (`save_checkpoint`)
- `scripts/train.py`
  - `argparse`로 config 경로 입력
  - YAML 로드
  - dataset/model 초기화
  - trainer 실행
- `scripts/sample.py`
  - checkpoint 로드
  - greedy decoding 기반 `generate` 구현
  - prompt 입력받아 텍스트 생성

## 설치

### (A) 기본 GPU 설치 (권장: cu128)

```powershell
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
py -3.11 -m pip install -r requirements.txt
```

### (B) 개발 도구까지 설치

```powershell
py -3.11 -m pip install -r requirements-dev.txt
```

### 설치 후 GPU 인식 확인

```powershell
py -3.11 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

### (C) fallback

```powershell
# 첫 fallback: cu126
py -3.11 -m pip install --upgrade --force-reinstall torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126

# 마지막 fallback: CPU
py -3.11 -m pip install --upgrade --force-reinstall torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

## 의존성 파일

- `requirements.txt` (runtime 최소)
  - `PyYAML>=6.0`
  - `tqdm>=4.66`
- `requirements-dev.txt` (선택)
  - `tensorboard>=2.17`
  - `rich>=13.7`
  - `black>=24.8`
  - `ruff>=0.6`
  - `pytest>=8.3`
  - `ipykernel>=6.29`

## 빠른 시작

### 1) 학습 텍스트 준비

`data/raw/input.txt` 파일을 생성하고 학습 텍스트를 넣습니다.
경로가 없으면 `data/raw/` 폴더를 먼저 생성한 뒤 파일을 만드세요.

### 2) 설정 파일 작성 (예: `configs/train.yaml`)

```yaml
data:
  text_path: data/raw/input.txt
  encoding: utf-8
  block_size: 128

model:
  max_seq_len: 128
  d_model: 256
  n_heads: 8
  n_layers: 6
  d_ff: 1024
  dropout: 0.1

train:
  batch_size: 32
  shuffle: true
  num_workers: 0
  drop_last: true
  lr: 3e-4
  weight_decay: 0.0
  betas: [0.9, 0.95]
  grad_clip: 1.0
  num_epochs: 5
  checkpoint_dir: checkpoints
  save_every: 500
```

### 3) 학습 실행

```powershell
py -3.11 scripts/train.py --config configs/train.yaml
```

### 4) 샘플 생성

```powershell
py -3.11 scripts/sample.py --checkpoint checkpoints/ckpt_epoch1_step500.pt --config configs/train.yaml --prompt "Hello" --max_new_tokens 100
```

프롬프트를 인자로 주지 않으면 실행 중 `prompt>` 입력을 받습니다.

## 체크포인트 형식

`Trainer.save_checkpoint()`는 아래 키를 저장합니다.

- `model_state_dict`
- `optimizer_state_dict`
- `config`
- `global_step`
- `epoch`
- `step`
