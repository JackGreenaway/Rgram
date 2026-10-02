"""Compare exact compact-neighbor and bounded brute-force prediction paths.

Run: uv run python examples/benchmark_neighbors.py
Numbers are workload/hardware dependent; fitting is timed separately.
"""

import time

import numpy as np

from rgram import KernelSmoother


def main() -> None:
    rng = np.random.default_rng(42)
    x = np.linspace(
        0, 100, 20000
    )  # Already sorted input is required for neighbor search.
    y = np.sin(x) + rng.normal(0, 0.1, len(x))
    query = np.linspace(1, 99, 500)
    model = KernelSmoother(kernel="tricube", bandwidth="manual", bandwidth_value=0.1)
    start = time.perf_counter()
    model.fit(x, y)
    print(f"Fit with already-sorted data: {time.perf_counter() - start:.4f}s")
    predictions = {}
    for algorithm in ("auto", "brute"):
        model.set_params(algorithm=algorithm)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            predictions[algorithm] = model.predict(query)
            times.append(time.perf_counter() - start)
        print(f"{algorithm}: median prediction time {np.median(times):.4f}s")
    np.testing.assert_allclose(predictions["auto"], predictions["brute"], atol=1e-12)
    print("Predictions agree to numerical tolerance.")


if __name__ == "__main__":
    main()
