import numpy as np
import math
from tqdm import tqdm
from collections.abc import Iterable, Callable

SCALES = [(1e12, "TFLOP"), (1e9, "GFLOP"), (1e6, "MFLOP"), (1e3, "KFLOP")]


def format_flops_range(a: float, b: float) -> tuple[str, str, str]:
    for scale, unit in SCALES:
        if a >= scale and b >= scale:
            return f"{a / scale:.2f}", f"{b / scale:.2f}", unit
    return f"{a:.0f}", f"{b:.0f}", "FLOP"


def plot_cost(elements: Iterable, cost_fn: Callable) -> None:
    costs = [cost_fn(e) for e in tqdm(elements)]
    counts, edges = np.histogram(costs, bins=32)
    widest = int(max(counts)) if len(counts) else 1
    for i, count in enumerate(counts):
        bar = "█" * math.ceil(60 * count / widest)
        left, right, unit = format_flops_range(edges[i], edges[i + 1])
        print(f"{left:>4}-{right:<4} {unit} │ {bar}")
