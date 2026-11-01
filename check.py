"""快速自检（几秒，不训练模型）。逐条验证论文的核心等式与本实现的约定。
运行 python check.py，全部通过才说明机制实现无误。"""
import itertools
import math
import random

import torch

import task
import unified as U


def test_weights():
    for n in (4, 8, 16):
        for k in range(1, n + 1):
            corr = [1] * k + [0] * (n - k)
            w = U.rl_weights(corr, "maxrl")
            assert abs(sum(w) - n) < 1e-9, "每题所有回答权重之和应固定为 N"
            assert all(x == 0 for x, c in zip(w, corr) if not c)
        assert U.rl_weights([0] * n) == [0.0] * n, "全错的题整体记零"
    assert U.rl_weights([1, 0, 1, 0], "reinforce") == [1.0, 0.0, 1.0, 0.0]
    print("ok  权重：MaxRL 每题权重和为 N，全错记零，REINFORCE 权重为 1")


def test_likelihood_identity():
    """论文 3.1：∇log p(x) = E_{y~π+}[∇log π(y)]。用 6 个结果的类别分布精确验证。"""
    torch.manual_seed(0)
    theta = torch.randn(6, requires_grad=True)
    reward = torch.tensor([1., 0., 1., 0., 0., 0.])
    pi = torch.softmax(theta, 0)
    p = (pi * reward).sum()
    g_lhs, = torch.autograd.grad(torch.log(p), theta, retain_graph=True)
    plus = (pi * reward / p).detach()
    g_rhs = torch.zeros(6)
    for y in range(6):
        gy, = torch.autograd.grad(torch.log(pi[y]), theta, retain_graph=True)
        g_rhs += plus[y] * gy
    assert torch.allclose(g_lhs, g_rhs, atol=1e-6)
    print("ok  恒等式：∇log p = E_{成功条件分布}[∇log π]")


def test_finite_n_expectation(n=3):
    """论文 4.2：有限 N 的估计量期望 = [1-(1-p)^N] · ∇log p。对全部 6^N 个结果精确枚举。"""
    torch.manual_seed(1)
    theta = torch.randn(6, requires_grad=True)
    reward = [1, 0, 1, 0, 0, 0]
    pi = torch.softmax(theta, 0)
    logpi = torch.log(pi)
    grads = [torch.autograd.grad(logpi[y], theta, retain_graph=True)[0] for y in range(6)]
    p = float(sum(pi[y] for y in range(6) if reward[y]).detach())
    expect = torch.zeros(6)
    for ys in itertools.product(range(6), repeat=n):
        prob = math.prod(float(pi[y]) for y in ys)
        corr = [reward[y] for y in ys]
        w = U.rl_weights(corr, "maxrl")
        est = sum(wi * grads[y] for wi, y in zip(w, ys)) / n
        expect += prob * est
    g_logp, = torch.autograd.grad(torch.log(sum(pi[y] for y in range(6) if reward[y])),
                                  theta)
    target = (1 - (1 - p) ** n) * g_logp
    assert torch.allclose(expect, target, atol=1e-5), (expect, target)
    print(f"ok  有限 N={n}：期望梯度 = [1-(1-p)^N]·∇log p，p={p:.3f}")


def test_mixture_sampler():
    rng = random.Random(0)
    short = lambda: U.lm_example([1] * 5)          # 4 个损失 token
    long_ = lambda: U.lm_example([1] * 21)         # 20 个损失 token
    s = U.MixtureSampler({"a": short, "b": long_}, {"a": 0.7, "b": 0.3}, rng)
    fr = s.empirical_token_fracs(30000)
    assert abs(fr["a"] - 0.7) < 0.02, fr
    count_ratio = s.p[0] / s.p[1]
    assert count_ratio > 3, "按条数比例应远大于 7:3，才能补偿长度差"
    print(f"ok  混合采样：目标 token 占比 0.7/0.3，实测 {fr['a']:.3f}/{fr['b']:.3f}，"
          f"条数比 {count_ratio:.1f}:1")


def test_loss_denominator():
    """全错回答（权重 0）的 token 仍计入分母。"""
    class Id(torch.nn.Module):
        def forward(self, ids):
            return torch.zeros(ids.size(0), ids.size(1), task.VOCAB, requires_grad=True)
    m = Id()
    prompt = [task.BOS, 3]
    good = U.rl_example(prompt, [5, 6, 2], 2.0)
    bad = U.rl_example(prompt, [5, 6, 2], 0.0)
    l_one = U.weighted_loss(m, [good]).item()
    l_two = U.weighted_loss(m, [good, bad]).item()
    assert abs(l_two - l_one / 2) < 1e-6
    print("ok  损失分母：权重为零的回答 token 仍计入 T_B")


def test_data_hygiene():
    train, held = set(task.TRAIN_STMTS), set(task.HELD_STMTS)
    assert not (train & held)
    frac = len(held) / len(task.ALL_STMTS)
    assert 0.05 < frac < 0.15
    ev = task.build_eval_sets(96)
    rng = random.Random(7)
    for kind in ("A", "B"):
        stream = task.RLStream(kind, [task.key(e) for e in ev[kind]], rng)
        seen = {task.key(stream.next()) for _ in range(3000)}
        assert not (seen & {task.key(e) for e in ev[kind]}), "RL 提示流不应含评测题"
    for combo in task.HELD_OUT:
        assert combo not in task.SEEN_COMBOS
    print(f"ok  数据：预训练语句 {len(task.ALL_STMTS)} 条，留出 {len(held)} 条且与训练集不相交；"
          f"RL 提示流与评测题去重；留出组合不进中训")


def test_schedule():
    import json
    from pipeline import Curve, Schedule, draw_mixed_batch
    c = Curve([[0, 1], [10, 1], [10, 0], [20, 0], [30, 1]])
    assert c(5) == 1 and c(10) == 0 and c(25) == 0.5 and c(40) == 1, "跳变与线性段"
    staged = Schedule(json.load(open("configs/staged.json")))
    for t in (0, 399, 400, 599, 600, 749, 750, 849):
        fr = staged.fractions(t)
        assert abs(sum(fr.values()) - 1) < 1e-9
        assert sorted(fr.values())[-1] == 1.0, f"分阶段在第 {t} 步应只有一个来源"
    uni = Schedule(json.load(open("configs/unified.json")))
    assert uni.fractions(300)["mid"] == 0.5 and uni.fractions(450)["rl_A"] == 0.85
    # 非 RL 步骤的混合：目标 token 占比 0.3 / 0.7
    rng = random.Random(0)
    short = lambda: U.lm_example([1] * 5)        # 4 个损失 token
    long_ = lambda: U.lm_example([1] * 21)       # 20 个损失 token
    T = {"a": 4.0, "b": 20.0}
    tok = {"a": 0, "b": 0}
    for _ in range(300):
        for ex in draw_mixed_batch({"a": 0.3, "b": 0.7}, {"a": short, "b": long_}, T, rng, 64):
            tok["a" if len(ex.ids) == 5 else "b"] += U.loss_tokens(ex)
    frac = tok["a"] / (tok["a"] + tok["b"])
    assert abs(frac - 0.3) < 0.02, frac
    print(f"ok  配比曲线：跳变与线性段正确；分阶段每步只有一个来源；混合步 token 占比 0.3，实测 {frac:.3f}")


if __name__ == "__main__":
    test_weights()
    test_likelihood_identity()
    test_finite_n_expectation()
    test_mixture_sampler()
    test_loss_denominator()
    test_data_hygiene()
    test_schedule()
    print("全部通过")
