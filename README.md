# Lazy MatMuls

Toy repo for lazily evaluating chains of matrix multiplies.
`uv sync` to install dependencies.

`uv run problem.py` to see the distribution of flops for random matrix multiplies.
`uv run benchmark.py` to see how much faster the optimal ordering is than a random ordering.