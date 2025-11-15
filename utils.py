import numpy as np
import math
from tqdm import tqdm
from collections.abc import Iterable, Callable

SCALES = [(1e12, "TFLOP"), (1e9, "GFLOP"), (1e6, "MFLOP"), (1e3, "KFLOP")]


def format_flops(flops: int) -> str:
    for scale, unit in SCALES:
        if flops >= scale:
            return f"{flops / scale:.2f} {unit}"
    return f"{flops:.0f} FLOP"


def plot_cost(elements: Iterable, cost_fn: Callable) -> None:
    costs = [cost_fn(e) for e in tqdm(elements)]
    counts, edges = np.histogram(costs, bins=32)
    widest = int(max(counts)) if len(counts) else 1
    for i, count in enumerate(counts):
        bar = "█" * math.ceil(60 * count / widest)
        left, right = format_flops(edges[i]), format_flops(edges[i + 1])
        print(f"{left:>4}-{right:<4} FLOP │ {bar}")
