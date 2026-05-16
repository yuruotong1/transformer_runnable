
import torch
from transformer_core import (
    make_model,
    subsequent_mask,
    Batch,
    load_tokenizers,
    load_vocab,
    create_dataloaders,
    tokenize,
)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def load_trained_model(vocab_src, vocab_tgt, model_path="multi30k_model_final.pt",
                        N=6, d_model=512, d_ff=2048, h=8):
    model = make_model(len(vocab_src), len(vocab_tgt), N=N, d_model=d_model, d_ff=d_ff, h=h)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()
    return model


def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = []
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print("Source Text (Input)        : " + " ".join(src_tokens).replace("\n", ""))
        print("Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", ""))

        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results.append((rb, src_tokens, tgt_tokens, model_out, model_txt))
    return results


def translate(text, model, vocab_src, vocab_tgt, spacy_de, max_len=72, pad_idx=2, eos_string="</s>"):
    """将一句德语文本翻译成英语，返回翻译结果字符串。"""
    tokens = ["<s>"] + tokenize(text, spacy_de) + ["</s>"]
    src_ids = [vocab_src[tok] if tok in vocab_src else vocab_src["<unk>"] for tok in tokens]
    src = torch.LongTensor(src_ids).unsqueeze(0)
    src_mask = (src != pad_idx).unsqueeze(-2)

    model_out = greedy_decode(model, src, src_mask, max_len, start_symbol=0)[0]
    translation = (
        " ".join([vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx])
        .split(eos_string, 1)[0]
        .strip()
    )
    return translation


def interactive_translate(model_path="multi30k_model_demo.pt"):
    """交互式翻译循环：持续读取德语输入，输出英语翻译，输入 'quit' 退出。"""
    print("Loading tokenizers...")
    spacy_de, spacy_en = load_tokenizers()

    print("Loading vocabulary...")
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    print("Loading trained model from:", model_path)
    model = load_trained_model(vocab_src, vocab_tgt, model_path, N=2, d_model=256, d_ff=1024, h=4)

    print("\nTranslator ready (German -> English). Type 'quit' to exit.\n")
    while True:
        text = input("Input (DE): ").strip()
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        result = translate(text, model, vocab_src, vocab_tgt, spacy_de)
        print(f"Output (EN): {result}\n")


def main(model_path="multi30k_model_final.pt", n_examples=5):
    print("Loading tokenizers...")
    spacy_de, spacy_en = load_tokenizers()

    print("Loading vocabulary...")
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    print("Loading trained model from:", model_path)
    model = load_trained_model(vocab_src, vocab_tgt, model_path,
                               N=2, d_model=256, d_ff=1024, h=4)

    print("Preparing validation data...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Checking model outputs:")
    check_outputs(valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples)




if __name__ == "__main__":
    interactive_translate("multi30k_model_demo_3.pt")
