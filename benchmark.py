import numpy as np
from lazy_matmul import optimal_order
from problem import flops, random_matrices, random_order
from tqdm import tqdm
from utils import format_flops

if __name__ == "__main__":
    matss = [random_matrices(100) for _ in range(10)]
    random_flops = [flops(random_order(mats)) for mats in tqdm(matss)]
    optimal_flops = [flops(optimal_order(mats)) for mats in tqdm(matss)]

    print(f"Random ordering: {format_flops(int(np.mean(random_flops)))}")
    print(f"Optimal ordering: {format_flops(int(np.mean(optimal_flops)))}")
    print(f"Speed-up: {np.mean(random_flops) / np.mean(optimal_flops):7.1f}×")
