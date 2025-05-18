from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from problem import Matrix, Parenthization, get_product
from functools import cache

def optimal_parenthization(matrices: list[Matrix]) -> tuple[int, Parenthization | Matrix]:
    n = len(matrices)
    if n == 1:
        return 0, matrices[0]

    dims = [matrices[0].shape[0]] + [m.shape[1] for m in matrices]
    mult_cost = lambda i, k, j : dims[i] * dims[k + 1] * dims[j + 1]

    @cache
    def solve(i: int, j: int) -> tuple[int, Parenthization | Matrix]:
        if i == j:
            return 0, matrices[i]

        best_cost, best_tree = float("inf"), None

        for k in range(i, j):
            left_cost,  left_tree  = solve(i, k)
            right_cost, right_tree = solve(k + 1, j)

            cost = mult_cost(i, k, j)

            total_cost = left_cost + right_cost + cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_tree = Parenthization(left_tree, right_tree)

        return best_cost, best_tree

    return solve(0, n - 1)


@dataclass
class LazyMat:
    matrices: list[Matrix] = field(default_factory=list)

    def __matmul__(self, other: Matrix | LazyMat) -> LazyMat:
        if isinstance(other, Matrix):
            return LazyMat(self.matrices + [other])

        if isinstance(other, LazyMat):
            return LazyMat(self.matrices + other.matrices)
        
        raise NotImplementedError

    def __rmatmul__(self, other: Matrix | LazyMat) -> LazyMat:
        if isinstance(other, Matrix):
            return LazyMat([other] + self.matrices)

        if isinstance(other, LazyMat):
            return LazyMat(other.matrices + self.matrices)
        
        raise NotImplementedError

    def value(self) -> Matrix:
        _, tree = optimal_parenthization(self.matrices)
        prod = get_product(tree)
        self.matrices = [prod]
        return prod

    def __array__(self, dtype=None):
        out = self.value()
        return np.asarray(out, dtype=dtype) if dtype is not None else out
