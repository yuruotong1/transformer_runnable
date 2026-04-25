import torch
import torch.nn as nn

from the_annotated_transformer import MultiHeadedAttention


def demo_1_shapes():
    print("=" * 70)
    print('Demo 1: feed the sentence "I love cats" into MultiHeadedAttention')
    print("=" * 70)

    sentence = ["I", "love", "cats"]
    vocab = {"<pad>": 0, "I": 1, "love": 2, "cats": 3, "dogs": 4}

    token_ids = torch.tensor([[vocab[w] for w in sentence]])
    print(f"sentence   = {sentence}")
    print(f"token_ids  = {token_ids}   shape={tuple(token_ids.shape)}   [batch=1, L=3]")

    h, d_model = 4, 16
    embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=d_model)

    x = embedding(token_ids)
    print(f"x = embedding(token_ids)   shape={tuple(x.shape)}   [batch=1, L=3, d_model=16]")

    mha = MultiHeadedAttention(h=h, d_model=d_model, dropout=0.0)
    mha.eval()

    out = mha(x, x, x)

    print(f"\noutput shape    = {tuple(out.shape)}   (same as input: [1, 3, 16])")
    print(f"mha.attn.shape  = {tuple(mha.attn.shape)}   [batch=1, heads=4, L_q=3, L_k=3]")

    print("\n--- attention weight for each of the 4 heads ---")
    print("(row = which word is asking, column = which word it attends to)")
    for head_idx in range(h):
        attn = mha.attn[0, head_idx].detach().round(decimals=3)
        print(f"\nhead {head_idx}:")
        print(f"           {'  '.join(f'{w:>5}' for w in sentence)}")
        for i, row_word in enumerate(sentence):
            row_vals = "  ".join(f"{v.item():>5.3f}" for v in attn[i])
            print(f"  {row_word:>5} -> {row_vals}")

    print("\n--- interpretation ---")
    print("Each row sums to 1.0 (it's a probability distribution).")
    print('Row "I" = how much "I" attends to each of "I", "love", "cats".')
    print("Because the model is untrained and the embeddings are random,")
    print("the distributions are basically noise -- but the mechanics work.")
    print("After training on real data, different heads would learn")
    print("meaningful patterns (subject-verb links, object tracking, etc).")


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_1_shapes()
