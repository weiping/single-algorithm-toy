"""极小的 decoder-only Transformer（对应论文 124M 基座的位置，约 80 万参数）。"""
import torch
import torch.nn as nn


class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=128, n_head=4, n_layer=4, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight  # 权重绑定
        self.max_len = max_len

    def forward(self, ids):
        T = ids.size(1)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        h = self.emb(ids) + self.pos.weight[:T]
        return self.head(self.ln(self.blocks(h, mask=mask)))

    @torch.no_grad()
    def generate(self, prompts, max_new, eos, temperature=1.0):
        """批量采样。prompts 为 list[list[int]]，返回只含生成部分的 list[list[int]]。
        按提示长度分组，组内等长，不需要填充。"""
        self.eval()
        out = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        groups = []
        for i in order:
            if groups and len(prompts[groups[-1][0]]) == len(prompts[i]):
                groups[-1].append(i)
            else:
                groups.append([i])
        for g in groups:
            cur = torch.tensor([prompts[i] for i in g], dtype=torch.long)
            B = len(g)
            alive = torch.ones(B, dtype=torch.bool)
            buf = torch.full((B, max_new), eos, dtype=torch.long)
            for t in range(max_new):
                logits = self.forward(cur)[:, -1]
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(-1)
                buf[:, t] = torch.where(alive, nxt, torch.full_like(nxt, eos))
                alive = alive & (nxt != eos)
                if not alive.any():
                    break
                cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
            for j, i in enumerate(g):
                row = buf[j].tolist()
                if eos in row:
                    row = row[:row.index(eos) + 1]
                out[i] = row
        return out
