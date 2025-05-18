from __future__ import annotations
import random
from dataclasses import dataclass
import numpy as np
from utils import performance_barplot
from functools import cache

Matrix = np.ndarray

@dataclass
class Parenthization:
    left: Parenthization | Matrix
    right: Parenthization | Matrix
    
    def __repr__(self) -> str:
        def _show(node: "Parenthization | Matrix") -> str:
            if isinstance(node, Matrix):
                return "M"
            return f"({_show(node.left)}, {_show(node.right)})"
        return _show(self)

def random_matrices(n: int) -> list[Matrix]:
    dims = [random.choice((1, 1_000)) for _ in range(n + 1)]
    return [np.random.randn(dims[i], dims[i + 1]) for i in range(n)]


def random_parenthization(matrices: list[Matrix]) -> Parenthization | Matrix:
    if len(matrices) == 1:
        return matrices[0]
    k = random.randint(1, len(matrices) - 1)
    left = random_parenthization(matrices[:k])
    right = random_parenthization(matrices[k:])
    return Parenthization(left, right)


def get_product(node: Parenthization | Matrix) -> Matrix:
    if isinstance(node, Matrix):
        return node
    return get_product(node.left) @ get_product(node.right)

if __name__ == '__main__':
    matrices = random_matrices(100)
    parenthizations = [random_parenthization(matrices) for _ in range(1_000)]
    performance_barplot(parenthizations, get_product)

