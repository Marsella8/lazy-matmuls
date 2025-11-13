from __future__ import annotations
from dataclasses import dataclass, field
from problem import Matrix, Matmul, flops
from functools import cache, cached_property


def optimal_order(matrices: list[Matrix]) -> Matmul | Matrix:
    n = len(matrices)
    if n == 1:
        return matrices[0]

    @cache
    def solve(i: int, j: int) -> Matmul | Matrix:
        if i == j:
            return matrices[i]
        trees = [Matmul(solve(i, k), solve(k + 1, j)) for k in range(i, j)]
        return min(trees, key=flops)

    return solve(0, n - 1)


@dataclass
class LazyMat:
    matrices: list[Matrix] = field(default_factory=list)

    def __matmul__(self, other: LazyMat) -> LazyMat:
        return LazyMat(self.matrices + other.matrices)

    def __rmatmul__(self, other: LazyMat) -> LazyMat:
        return self @ other

    @cached_property
    def value(self) -> Matrix:
        tree = optimal_order(self.matrices)
        if isinstance(tree, Matrix):
            return tree
        return tree.value()
