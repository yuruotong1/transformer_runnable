"""
Transformer 训练入口脚本
使用 .conda/python.exe train.py 运行
"""

from the_annotated_transformer import (
    load_tokenizers,
    load_vocab,
    train_model,
    make_model,
    greedy_decode,
    Batch,
    create_dataloaders,
    check_outputs,
)
import torch


def main():
    # 1. 加载分词器
    print("=" * 50)
    print("Step 1: Loading tokenizers...")
    spacy_de, spacy_en = load_tokenizers()
    print("Done.")

    # 2. 加载/构建词表
    print("=" * 50)
    print("Step 2: Loading vocabulary...")
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    print("Done.")

    # 3. 训练配置
    config = {
        "batch_size": 32,
        "distributed": False,      # CPU 单卡训练，设为 False
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }

    # 4. 开始训练
    print("=" * 50)
    print("Step 3: Starting training...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print("=" * 50)

    train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    print("=" * 50)
    print("Training completed!")
    print("Model saved as: multi30k_model_final.pt")


if __name__ == "__main__":
    main()
