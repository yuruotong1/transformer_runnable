"""
Transformer 训练入口脚本
使用 .conda/python.exe train.py 运行

两种模式：
  train_demo()  —— 缩小模型 + 少量数据，CPU 约 3~5 分钟，用于演示
  train_full()  —— 原始论文配置，完整训练
"""

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from transformer_core import (
    load_tokenizers,
    load_vocab,
    train_model,
    make_model,
    run_epoch,
    LabelSmoothing,
    SimpleLossCompute,
    TrainState,
    rate,
    Batch,
    collate_batch,
    tokenize,
    load_multi30k_raw,
    create_dataloaders,
)

try:
    from torchtext import datasets
    _USE_TORCHTEXT = True
except Exception:
    _USE_TORCHTEXT = False


# ---------------------------------------------------------------------------
# 演示模式：小模型 + 少量数据，CPU 约 3~5 分钟
# ---------------------------------------------------------------------------

def train_demo(vocab_src, vocab_tgt, spacy_de, spacy_en):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 缩小模型（原始论文: N=6, d_model=512, d_ff=2048, h=8）
    N, d_model, d_ff, h = 2, 256, 1024, 4
    model = make_model(len(vocab_src), len(vocab_tgt), N=N, d_model=d_model, d_ff=d_ff, h=h)
    model.to(device)

    pad_idx = vocab_tgt["<blank>"]
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.to(device)

    optimizer = Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    warmup = 400
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=warmup),
    )

    # 加载数据，只取前 3000 句
    def _tokenize_de(text): return tokenize(text, spacy_de)
    def _tokenize_en(text): return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch, _tokenize_de, _tokenize_en, vocab_src, vocab_tgt,
            device, max_padding=32, pad_id=vocab_src.get_stoi()["<blank>"],
        )

    
    train_iter, valid_iter, _ = datasets.Multi30k(language_pair=("de", "en"))
    train_data = list(train_iter)[:3000]
    valid_data = list(valid_iter)[:500]
  
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_state = TrainState()
        print(f"\n[Demo] Epoch {epoch + 1}/{num_epochs}")
        run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=4,
            train_state=train_state,
        )
        model.eval()
        with torch.no_grad():
            val_loss, _ = run_epoch(
                (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
                model,
                SimpleLossCompute(model.generator, criterion),
                DummyOptimizer(),
                DummyScheduler(),
                mode="eval",
            )
        print(f"  Val loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "multi30k_model_demo.pt")
    print("\nDemo training done. Saved: multi30k_model_demo.pt")


# ---------------------------------------------------------------------------
# 完整训练：原始论文配置
# ---------------------------------------------------------------------------

def train_full(vocab_src, vocab_tgt, spacy_de, spacy_en):
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Epochs: {config['num_epochs']}, Batch: {config['batch_size']}")
    train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    print("Full training done. Saved: multi30k_model_final.pt")


# ---------------------------------------------------------------------------
# 占位用的 dummy 优化器（eval 时不更新参数）
# ---------------------------------------------------------------------------

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        self.state = {}
    def step(self): pass
    def zero_grad(self, set_to_none=False): pass

class DummyScheduler:
    def step(self): pass


# ---------------------------------------------------------------------------

def main():
    print("Step 1: Loading tokenizers...")
    spacy_de, spacy_en = load_tokenizers()

    print("Step 2: Loading vocabulary...")
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    # 切换这里选择模式
    MODE = "demo"   # "demo" 或 "full"

    print(f"Step 3: Training ({MODE} mode)...")
    if MODE == "demo":
        train_demo(vocab_src, vocab_tgt, spacy_de, spacy_en)
    else:
        train_full(vocab_src, vocab_tgt, spacy_de, spacy_en)


if __name__ == "__main__":
    main()
