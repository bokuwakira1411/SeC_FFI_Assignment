"""I1 student: predicts the 256-d SeC concept vector from SAM 2 (Hiera) features, without the LLM.

Input per sample: up to 7 keyframes (start frame + up to 6 others, as the MLLM memory given to the LLM) and the
current frame. Each frame = 16x16 tokens of the 256-d image feature (current_vision_feats[-1], avg-pooled 4x4).
Keyframe tokens get a mask channel (GT / predicted mask area fraction at 16x16); the current frame gets 0.
Token = Linear(257 -> 256) + 2D position embedding + slot embedding
        (0 = start frame, 1..6 = keyframes by recency (1 = most recent), 7 = current frame).
Decoder: 1 learned query, 2 layers of (cross-attention to all tokens + FFN), d=256, 8 heads, FFN 1024,
then Linear -> 256 (standardised space); the output is de-standardised with the teacher train statistics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_KEY = 7          # MLLM memory given to the LLM: start frame + 6 most recent
CUR_SLOT = 7


class Layer(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.nq, self.nkv, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, q, kv, pad):
        kv = self.nkv(kv)
        q = q + self.attn(self.nq(q), kv, kv, key_padding_mask=pad, need_weights=False)[0]
        return q + self.ffn(self.n2(q))


class ConceptStudent(nn.Module):
    def __init__(self, d=256, heads=8, ffn=1024, layers=2, feat_dim=256, out_dim=256):
        super().__init__()
        self.inp = nn.Linear(feat_dim + 1, d)
        self.pos = nn.Parameter(torch.randn(256, d) * 0.02)          # 16x16 positions
        self.slot = nn.Embedding(N_KEY + 1, d)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList(Layer(d, heads, ffn) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, out_dim)
        self.register_buffer("mu", torch.zeros(out_dim))
        self.register_buffer("sigma", torch.ones(out_dim))

    def forward(self, feats, masks, slots, valid):
        """feats [B, F, 256, 16, 16], masks [B, F, 16, 16] (0 for the current frame), slots [B, F] long,
        valid [B, F] bool. Returns the standardised prediction [B, 256]."""
        B, Fn = slots.shape
        x = torch.cat([feats.flatten(3), masks.flatten(2).unsqueeze(2)], 2)    # [B, F, 257, 256]
        x = self.inp(x.transpose(2, 3))                                          # [B, F, 256 tok, d]
        x = x + self.pos + self.slot(slots)[:, :, None]
        x = x.reshape(B, Fn * 256, -1)
        pad = (~valid)[:, :, None].expand(B, Fn, 256).reshape(B, Fn * 256)
        q = self.query.expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, x, pad)
        return self.out(self.norm(q[:, 0]))

    def concept(self, feats, masks, slots, valid):
        """De-standardised concept vector [B, 256] (what token_attn receives)."""
        return self.forward(feats, masks, slots, valid) * self.sigma + self.mu


def slots_for(n_keys):
    """Slots for n_keys keyframes in time order (first = start frame) followed by the current frame."""
    return [0] + [n_keys - i for i in range(1, n_keys)] + [CUR_SLOT]
