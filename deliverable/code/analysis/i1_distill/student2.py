"""I1 fixes (after stop 2): small object-centric student (1) and hybrid student (3).

Tokens (d = 128) from the 16x16 Hiera features (256-d):
  keyframe:      masked average of the features over the mask (mask area fraction as weights) + global average,
                 plus the mask area -> 2 tokens per keyframe;
  current frame: 4x4 average-pooled features + global average -> 17 tokens, no mask;
  anchor frame (hybrid only): same 17 tokens as the current frame, its own type embedding;
  anchor vector (hybrid only): the last LLM concept (standardised) -> 1 token.
Each token gets a slot/type embedding (0 = start frame, 1..6 = keyframes by recency, 7 = current, 8 = anchor frame,
9 = anchor vector) and, for 4x4 tokens, a position embedding.
Decoder: 1 learned query, 2 layers (cross-attention + FFN), d=128, 4 heads, FFN 256, dropout 0.1, -> Linear 256.
Hybrid: the output is a residual added to the standardised anchor vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_KEY = 7
CUR, ANC_F, ANC_V = 7, 8, 9


class Layer(nn.Module):
    def __init__(self, d, heads, ffn, drop):
        super().__init__()
        self.nq, self.nkv, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, d))
        self.drop = nn.Dropout(drop)

    def forward(self, q, kv, pad):
        kv = self.nkv(kv)
        q = q + self.drop(self.attn(self.nq(q), kv, kv, key_padding_mask=pad, need_weights=False)[0])
        return q + self.drop(self.ffn(self.n2(q)))


def frame_tokens(feats):
    """feats [..., 256, 16, 16] -> [..., 17, 256] (4x4 pooled + global)."""
    sh = feats.shape[:-3]
    x = feats.reshape(-1, 256, 16, 16).float()
    p = F.avg_pool2d(x, 4).flatten(2).transpose(1, 2)                  # [N, 16, 256]
    g = x.mean((2, 3))[:, None]                                         # [N, 1, 256]
    return torch.cat([p, g], 1).reshape(*sh, 17, 256)


def key_tokens(feats, masks):
    """feats [..., 256, 16, 16], masks [..., 16, 16] -> [..., 2, 257] (masked mean | area, global mean | area)."""
    x = feats.float().flatten(-2)                                       # [..., 256, 256]
    w = masks.float().flatten(-2)                                       # [..., 256]
    area = w.mean(-1, keepdim=True)
    obj = (x * w[..., None, :]).sum(-1) / w.sum(-1, keepdim=True).clamp_min(1e-6)
    glob = x.mean(-1)
    return torch.stack([torch.cat([obj, area], -1), torch.cat([glob, area], -1)], -2)


class ObjStudent(nn.Module):
    def __init__(self, hybrid=False, d=128, heads=4, ffn=256, layers=2, drop=0.1, out_dim=256):
        super().__init__()
        self.hybrid = hybrid
        self.key_in = nn.Linear(257, d)
        self.frm_in = nn.Linear(256, d)
        self.kind = nn.Embedding(2, d)                                  # keyframe: object token / global token
        self.pos = nn.Parameter(torch.randn(17, d) * 0.02)               # 4x4 + global
        self.slot = nn.Embedding(10, d)
        if hybrid:
            self.vec_in = nn.Linear(out_dim, d)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList(Layer(d, heads, ffn, drop) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, out_dim)
        if hybrid:
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # starts as "copy the anchor"
        self.register_buffer("mu", torch.zeros(out_dim))
        self.register_buffer("sigma", torch.ones(out_dim))

    def forward(self, key_f, key_m, key_slots, key_valid, cur_f, anc_f=None, anc_z=None):
        """key_f [B, 7, 256, 16, 16], key_m [B, 7, 16, 16], key_slots [B, 7], key_valid [B, 7], cur_f [B, 256, 16, 16],
        anc_f [B, 256, 16, 16], anc_z [B, 256] (standardised anchor vector). Returns standardised prediction [B, 256]."""
        B = cur_f.shape[0]
        k = self.key_in(key_tokens(key_f, key_m)) + self.kind.weight + self.slot(key_slots)[:, :, None]  # [B,7,2,d]
        toks = [k.reshape(B, -1, k.shape[-1])]
        pads = [(~key_valid)[:, :, None].expand(B, N_KEY, 2).reshape(B, -1)]
        c = self.frm_in(frame_tokens(cur_f)) + self.pos + self.slot.weight[CUR]
        toks.append(c); pads.append(torch.zeros(B, 17, dtype=torch.bool, device=c.device))
        if self.hybrid:
            a = self.frm_in(frame_tokens(anc_f)) + self.pos + self.slot.weight[ANC_F]
            v = self.vec_in(anc_z)[:, None] + self.slot.weight[ANC_V]
            toks += [a, v]
            pads += [torch.zeros(B, 18, dtype=torch.bool, device=c.device)]
        x, pad = torch.cat(toks, 1), torch.cat(pads, 1)
        q = self.query.expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, x, pad)
        z = self.out(self.norm(q[:, 0]))
        return z + anc_z if self.hybrid else z

    def concept(self, *args, **kw):
        return self.forward(*args, **kw) * self.sigma + self.mu
