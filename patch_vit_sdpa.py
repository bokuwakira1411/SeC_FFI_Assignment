# Replace InternViT naive attention (materializes full NxN matrix) with PyTorch SDPA
# (memory-efficient kernel, same math). Run from SeC/.
p = "inference/modeling_intern_vit.py"
s = open(p).read()
old = ("        attn = ((q * self.scale) @ k.transpose(-2, -1))\n"
       "        attn = attn.softmax(dim=-1)\n"
       "        attn = self.attn_drop(attn)\n\n"
       "        x = (attn @ v).transpose(1, 2).reshape(B, N, C)\n")
new = ("        # patched: SDPA instead of explicit NxN attention matrix (saves VRAM)\n"
       "        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)\n"
       "        x = x.transpose(1, 2).reshape(B, N, C)\n")
assert "patched: SDPA" not in s, "already patched"
assert s.count(old) == 1, "pattern not found"
s = s.replace(old, new)
if "import torch.nn.functional as F" not in s:
    s = s.replace("import torch\n", "import torch\nimport torch.nn.functional as F\n", 1)
open(p, "w").write(s)
print("patched")
