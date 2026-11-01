"""统一训练的核心机制（对应论文第 3、4 节）。

1. 样本是 (z, w)。z 是 token 序列，w 是逐 token 的损失权重，resp 标出非提示 token。
   预训练 / 中训权重为 1，SFT 提示部分为 0，RL 回答部分乘上奖励权重（式 2 的数据格式）。
2. 一个损失函数处理所有来源：权重乘 NLL 求和，除以批内非提示 token 总数 T_B。
   权重为零的全错回答的 token 也计入分母（论文附录 C.3）。
3. RL 权重是 MaxRL 的有限样本形式：每题 N 条，正确回答权重 1/p̂，错误为 0。
   objective="reinforce" 时正确回答权重取 1（论文 Table 2 只差有效提示分布）。
4. 混合采样速率正比于 f_s / T_s，T_s 是该来源每个样本的平均「损失 token」数。
5. 生成与训练解耦：生成器用落后的权重副本，每 sync_every 次更新同步一次，
   用单进程模拟论文的异步训练（第 4.3 节、附录 D.6）。
"""
import copy

import torch
import torch.nn.functional as F

import task


# ---------------- 数据格式 ----------------
class Example:
    __slots__ = ("ids", "w", "resp")

    def __init__(self, ids, w, resp):
        self.ids, self.w, self.resp = ids, w, resp


def lm_example(ids):
    """预训练 / 中训：全部 token 权重为 1。"""
    return Example(ids, [1.0] * len(ids), [True] * len(ids))


def sft_example(prompt, response):
    n = len(prompt)
    return Example(prompt + response, [0.0] * n + [1.0] * len(response),
                   [False] * n + [True] * len(response))


def rl_example(prompt, gen, weight):
    n = len(prompt)
    return Example(prompt + gen, [0.0] * n + [weight] * len(gen),
                   [False] * n + [True] * len(gen))


def loss_tokens(ex):
    """该样本贡献给分母 T_B 的 token 数（位置 0 不被预测，不计）。"""
    return sum(ex.resp[1:])


def weighted_loss(model, examples):
    """式 (2)：-(1/T_B) Σ w·log π，T_B 为批内非提示 token 总数。"""
    maxlen = max(len(e.ids) for e in examples)
    B = len(examples)
    ids = torch.full((B, maxlen), task.PAD, dtype=torch.long)
    w = torch.zeros(B, maxlen)
    resp = torch.zeros(B, maxlen)
    for i, e in enumerate(examples):
        n = len(e.ids)
        ids[i, :n] = torch.tensor(e.ids)
        w[i, :n] = torch.tensor(e.w)
        resp[i, :n] = torch.tensor([float(r) for r in e.resp])
    logp = F.log_softmax(model(ids)[:, :-1], dim=-1)
    nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    denom = resp[:, 1:].sum().clamp(min=1.0)
    return (nll * w[:, 1:]).sum() / denom


# ---------------- RL 权重 ----------------
def rl_weights(correct, objective="maxrl"):
    """correct 为同一道题 N 条回答的 0/1 列表。全错的题整体记零。"""
    n, k = len(correct), sum(correct)
    if k == 0:
        return [0.0] * n
    a = n / k if objective == "maxrl" else 1.0       # n/k = 1/p̂
    return [a if c else 0.0 for c in correct]


# ---------------- 混合采样 ----------------
class MixtureSampler:
    """按目标 token 占比 f_s 混合多个来源。采样速率 ∝ f_s / T_s。"""

    def __init__(self, sources, token_fracs, rng, probe=300):
        self.sources, self.rng = sources, rng
        self.T = {}
        for name, fn in sources.items():
            self.T[name] = sum(loss_tokens(fn()) for _ in range(probe)) / probe
        rates = {s: token_fracs[s] / self.T[s] for s in sources}
        tot = sum(rates.values())
        self.names = list(sources)
        self.p = [rates[s] / tot for s in self.names]

    def draw(self):
        return self.sources[self.rng.choices(self.names, weights=self.p)[0]]()

    def empirical_token_fracs(self, n=20000):
        toks = {s: 0.0 for s in self.names}
        for _ in range(n):
            s = self.rng.choices(self.names, weights=self.p)[0]
            toks[s] += loss_tokens(self.sources[s]())
        tot = sum(toks.values())
        return {s: toks[s] / tot for s in toks}


# ---------------- 算力记账 ----------------
class Meter:
    """粗略 FLOPs：训练 6·P·token，生成 2·P·token（按带 KV 缓存的理想情形）。"""

    def __init__(self, n_params):
        self.P, self.flops, self.train_tokens, self.gen_tokens = n_params, 0.0, 0, 0

    def train(self, examples):
        n = sum(len(e.ids) for e in examples)
        self.train_tokens += n
        self.flops += 6 * self.P * n

    def gen(self, n_tokens):
        self.gen_tokens += n_tokens
        self.flops += 2 * self.P * n_tokens


# ---------------- 训练循环 ----------------
def step(model, opt, examples, meter):
    model.train()
    loss = weighted_loss(model, examples)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    meter.train(examples)
    return loss.item()


def likelihood_phase(model, opt, batch_fn, steps, meter, tag="", log_every=100):
    """预训练 / 中训 / SFT 共用的阶段：只是 batch_fn 给出的数据不同。"""
    for s in range(steps):
        loss = step(model, opt, batch_fn(), meter)
        if log_every and (s + 1) % log_every == 0:
            print(f"    [{tag}] step {s + 1}/{steps} loss {loss:.3f}", flush=True)


def generate_rl_batch(gen_model, exprs, n, objective, meter):
    """用（可能落后的）生成器模型采样，返回带权重的样本与统计。"""
    prompts = [[task.BOS] + task.tokenize_expr(e) for e in exprs]
    flat = [p for p in prompts for _ in range(n)]
    gens = gen_model.generate(flat, max_new=8, eos=task.EOS)
    meter.gen(sum(len(p) + len(g) for p, g in zip(flat, gens)))
    examples, p_sum, any_ok = [], 0.0, 0
    for gi, e in enumerate(exprs):
        group = gens[gi * n:(gi + 1) * n]
        target = task.expr_value(e)
        corr = [int(task.is_correct(g, target)) for g in group]
        p_sum += sum(corr) / n
        any_ok += int(sum(corr) > 0)
        for g, a in zip(group, rl_weights(corr, objective)):
            examples.append(rl_example(prompts[gi], g, a))
    return examples, {"p_hat": p_sum / len(exprs), "any_ok": any_ok / len(exprs)}


def rl_phase(model, opt, cfg, stream, meter, updates, demo_src=None, demo_frac=0.0,
             tag="rl", log_every=50):
    """MaxRL / REINFORCE 阶段。可按回答 token 占比 demo_frac 混入示范数据。
    生成器是 model 的副本，每 cfg.sync_every 次更新同步一次权重。"""
    gen_model = copy.deepcopy(model)
    last_sync, stale, recent = 0, [], []
    for u in range(updates):
        if u % cfg.sync_every == 0:
            gen_model.load_state_dict(model.state_dict())
            last_sync = u
        stale.append(u - last_sync)
        exprs = [stream.next() for _ in range(cfg.rl_prompts)]
        examples, st = generate_rl_batch(gen_model, exprs, cfg.rl_n, cfg.objective, meter)
        recent.append(st["p_hat"])
        if demo_src is not None and demo_frac > 0:
            rl_tok = sum(loss_tokens(e) for e in examples)
            want, got = demo_frac / (1 - demo_frac) * rl_tok, 0
            while got < want:
                ex = demo_src()
                examples.append(ex)
                got += loss_tokens(ex)
        loss = step(model, opt, examples, meter)
        if log_every and (u + 1) % log_every == 0:
            print(f"    [{tag}] update {u + 1}/{updates} loss {loss:.3f} "
                  f"训练通过率 {sum(recent[-log_every:]) / len(recent[-log_every:]):.3f}",
                  flush=True)
    return {"mean_staleness": sum(stale) / len(stale)}


# ---------------- 评测 ----------------
@torch.no_grad()
def heldout_loss(model, stmts):
    """留出语句上的每 token 负对数似然（真正留出：这些语句从未进入训练）。"""
    model.eval()
    exs = [lm_example(task.stmt_tokens(s)) for s in stmts]
    tot, cnt = 0.0, 0
    for i in range(0, len(exs), 128):
        chunk = exs[i:i + 128]
        n = sum(loss_tokens(e) for e in chunk)
        tot += weighted_loss(model, chunk).item() * n
        cnt += n
    return tot / cnt


@torch.no_grad()
def pass_at_k(model, exprs, k=16):
    prompts = [[task.BOS] + task.tokenize_expr(e) for e in exprs]
    flat = [p for p in prompts for _ in range(k)]
    gens = model.generate(flat, max_new=8, eos=task.EOS)
    ok = []
    for i, e in enumerate(exprs):
        t = task.expr_value(e)
        ok.append([int(task.is_correct(g, t)) for g in gens[i * k:(i + 1) * k]])
    p1 = sum(sum(r) for r in ok) / (len(ok) * k)
    pk = sum(1 for r in ok if any(r)) / len(ok)
    return p1, pk
