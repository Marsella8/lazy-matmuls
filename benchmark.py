from __future__ import annotations
import time
import random
import numpy as np
from tqdm import tqdm
from lazy_matmul import LazyMat
from problem import random_matrices, Matrix


def random_matrices(n: int) -> list[Matrix]:
    dims = [random.randint(10, 1000) for _ in range(n + 1)]
    return [np.random.randn(dims[i], dims[i + 1]) for i in range(n)]

def naive_product(chain):
    """(((A @ B) @ C) @ …)."""
    out = chain[0]
    for m in chain[1:]:
        out = out @ m
    return out

def lazy_product(chain):
    """Lazy Matmul"""
    acc = LazyMat()
    for m in chain:
        acc = acc @ m
    return acc.value()

def bench(fn, mats, runs=10):
    t0 = time.perf_counter()
    for _ in tqdm(range(runs)):
        fn(mats)
    return (time.perf_counter() - t0) / runs

if __name__ == "__main__":
    random.seed(0); np.random.seed(0)

    mats = random_matrices(100)

    t_naive = bench(naive_product, mats)
    t_lazy  = bench(lazy_product,  mats)

    print(f"Naïve left‑to‑right : {t_naive*1e3:7.1f} ms")
    print(f"Lazy optimal        : {t_lazy*1e3:7.1f} ms")
    print(f"Speed‑up            : {t_naive/t_lazy:7.1f}×")
