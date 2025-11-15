from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from functools import cached_property
import random
from utils import plot_cost

np.random.seed(0)
random.seed(0)


Matrix = np.ndarray


@dataclass
class MatMul:
    left: MatMul | Matrix
    right: MatMul | Matrix

    @cached_property
    def shape(self) -> tuple[int, int]:
        (m, _), (_, p) = self.left.shape, self.right.shape
        return (m, p)

    def value(self) -> Matrix:
        def product(node: MatMul | Matrix) -> Matrix:
            if isinstance(node, Matrix):
                return node
            return product(node.left) @ product(node.right)

        return product(self)


def random_matrices(n: int) -> list[Matrix]:
    dims = [random.randint(10, 1000) for _ in range(n + 1)]
    return [np.random.randn(dims[i], dims[i + 1]) for i in range(n)]


def random_order(matrices: list[Matrix]) -> MatMul | Matrix:
    if len(matrices) == 1:
        return matrices[0]
    k = random.randint(1, len(matrices) - 1)
    left, right = random_order(matrices[:k]), random_order(matrices[k:])
    return MatMul(left, right)


def flops(node: MatMul | Matrix) -> int:
    if isinstance(node, Matrix):
        return 0
    (m, n), (n, p) = node.left.shape, node.right.shape
    return flops(node.left) + flops(node.right) + m * p * (2 * n - 1)


if __name__ == "__main__":
    matrices = random_matrices(100)
    orders = [random_order(matrices) for _ in range(10_000)]
    plot_cost(orders, flops)
