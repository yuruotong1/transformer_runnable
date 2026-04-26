import torch

from the_annotated_transformer import attention, subsequent_mask


def demo_1_minimal():
    print("=" * 60)
    print("Demo 1: minimal 2D case  shape=[L, d_k]")
    print("=" * 60)
    L, d_k = 4, 8
    torch.manual_seed(0)
    q = torch.randn(L, d_k)
    k = torch.randn(L, d_k)
    v = torch.randn(L, d_k)
    # mask = torch.ones(4, 4)
    # print(f"mask.shape = {mask.size(0)}")
    mask = subsequent_mask(4)
    out, attn = attention(q, k, v, mask=mask)

    print(f"q.shape    = {tuple(q.shape)}")
    print(f"out.shape  = {tuple(out.shape)}   (same as v)")
    print(f"attn.shape = {tuple(attn.shape)}  (square [L, L])")
    print(f"attn row sums (should all be 1.0) = {attn.sum(dim=-1)}")
    print("attention weight matrix:")
    print(attn.round(decimals=3))



if __name__ == "__main__":
    demo_1_minimal()
   
