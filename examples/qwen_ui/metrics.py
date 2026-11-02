"""评测指标。pass@k 用 Chen et al. (2021) 的无偏估计，论文每题采 64 条后据此算 pass@1 与 pass@16。"""
import math
import random


def pass_at_k(n, c, k):
    """n 条采样里 c 条正确时，k 次尝试至少对一次的概率的无偏估计。"""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def bootstrap_ci(values, reps=2000, seed=0, alpha=0.05):
    """按题重采样的均值置信区间。题目数少时区间会很宽，这正是需要报告它的原因。"""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(reps * alpha / 2)], means[int(reps * (1 - alpha / 2)) - 1]
