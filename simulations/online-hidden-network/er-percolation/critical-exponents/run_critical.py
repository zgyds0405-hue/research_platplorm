"""Paired critical-window simulations of ER site percolation and Markov node dynamics."""
from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent
scan_spec = spec_from_file_location("er_component_scan", ROOT.parent / "finite-size" / "simulate.py")
scan_module = module_from_spec(scan_spec)
scan_spec.loader.exec_module(scan_module)
compile_scan, check_scan, measure = scan_module.compile_scan, scan_module.check_scan, scan_module.measure

SIZES = np.array([1000, 4000, 16000, 64000, 256000, 1024000])
DEGREES = np.array([2, 4, 8])
REPLICATES = 40
SNAPSHOTS = 8
SEED = 4729477
MU = 0.01
WINDOW = np.linspace(-4, 4, 33)
POWER_GRID = np.geomspace(.012, .30, 41)


def probabilities(n, k):
    reduced = np.unique(np.r_[WINDOW / np.cbrt(n / k), POWER_GRID])
    return (1 + reduced) / k


def er_adjacency(n, k, rng):
    """Sample independent Bernoulli edges using geometric skips in one triangle."""
    total = n * (n - 1) // 2
    expected = n * k // 2
    block = expected + 12 * int(np.sqrt(expected)) + 32
    q = k / (n - 1)
    positions = -1 + np.cumsum(rng.geometric(q, size=block))
    while positions[-1] < total - 1:
        positions = np.r_[positions, positions[-1] + np.cumsum(rng.geometric(q, size=block))]
    positions = positions[:np.searchsorted(positions, total)]
    row = ((1 + np.sqrt(1 + 8 * positions)) / 2).astype(np.int64)
    col = positions - row * (row - 1) // 2
    return csr_matrix((np.ones(2 * len(row), dtype=np.int8),
                       (np.r_[row, col], np.r_[col, row])), shape=(n, n))


def run_group(task):
    k, n = map(int, task)
    scan = compile_scan()
    probs = probabilities(n, k)
    sums = np.zeros((REPLICATES, len(probs), 4), dtype=np.int32)
    for rep in range(REPLICATES):
        graph_rng, static_rng, markov_rng = map(np.random.default_rng,
            np.random.SeedSequence([SEED, k, n, rep]).spawn(3))
        adjacency = er_adjacency(n, k, graph_rng)
        offsets, neighbors = adjacency.indptr, adjacency.indices
        # A common uniform rank across the probability grid preserves every
        # marginal experiment. Zero ranks start the Markov chain all online.
        ranks = np.zeros(n)
        for i in range(SNAPSHOTS):
            sums[rep, :, :2] += measure(scan, offsets, neighbors, static_rng.random(n), probs)
            refresh = markov_rng.random(n) >= np.exp(-15 if i == 0 else -3)
            ranks[refresh] = markov_rng.random(np.count_nonzero(refresh))
            sums[rep, :, 2:] += measure(scan, offsets, neighbors, ranks, probs)
    print(f"k={k}, N={n:,}: {REPLICATES} paired graphs", flush=True)
    return f"k{k}_n{n}", sums


def main():
    check_scan(compile_scan())
    tasks = [(k, n) for k in DEGREES for n in SIZES]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(run_group, tasks))
    np.savez_compressed(ROOT / "paired_runs.npz", **results, sizes=SIZES, degrees=DEGREES,
                        replicates=REPLICATES, snapshots=SNAPSHOTS, seed=SEED, mu=MU,
                        window=WINDOW, power_grid=POWER_GRID)


if __name__ == "__main__":
    main()
