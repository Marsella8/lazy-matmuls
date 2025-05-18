import time
import numpy as np
import math
from tqdm import tqdm
from typing import Iterable, Callable

def performance_barplot(elements: Iterable, func : Callable) -> None:
    timings = []
    for p in tqdm(elements):
        start = time.perf_counter()
        func(p)
        timings.append(time.perf_counter() - start)

    bins = 16
    counts, edges = np.histogram(timings, bins=bins)
    widest = max(counts)
    for i, c in enumerate(counts):
        bar = "█" * math.ceil(40 * c / widest)
        print(f"{round(edges[i]*1e3, -1):6.0f}-{round(edges[i+1]*1e3, -1):6.0f} ms │ {bar}")
