import random
import numpy as np
from lazy_matmul import optimal_order
from problem import flops, random_matrices, random_order


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    mats = random_matrices(100)
    random_flops = flops(random_order(mats))
    optimal_flops = flops(optimal_order(mats))

    print(f"Random ordering: {random_flops:7.1f} flops")
    print(f"Optimal ordering: {optimal_flops:7.1f} flops")
    print(f"Speed-up: {random_flops / optimal_flops:7.1f}×")
