"""Coupled probability scans of static site percolation and dynamic snapshots."""
import ctypes
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent
SIZES = (2000, 4000, 8000, 16000, 32000)
DEGREES = (2, 4, 8)
GRAPH_COUNT = 10
SNAPSHOTS = 8
SEED = 487219
MU = .01
PROBS = np.arange(10, 991) / 1000


def compile_scan():
    with TemporaryDirectory() as build:
        lib = Path(build) / "components.so"
        subprocess.run(["g++", "-O3", "-std=c++17", "-fPIC", "-shared",
                        str(ROOT / "components.cpp"), "-o", str(lib)], check=True)
        scan = ctypes.CDLL(str(lib)).component_scan
    arr = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    scan.argtypes = [ctypes.c_int, arr, arr, arr, ctypes.c_int, arr, arr]
    scan.restype = None
    return scan


def measure(scan, offsets, neighbors, ranks, probs=PROBS):
    order = np.argsort(ranks).astype(np.int32)
    counts = np.searchsorted(ranks[order], probs, side="left").astype(np.int32)
    components = np.empty((len(probs), 2), dtype=np.int32)
    scan(len(ranks), offsets, neighbors, order, len(probs), counts, components)
    return components


def check_scan(scan):
    # Independent BFS checks include isolated vertices, tied sizes and empty graphs.
    rng = np.random.default_rng(199)
    for n in (2, 12, 53):
        g = nx.gnp_random_graph(n, min(3 / n, 1), seed=n)
        adj = nx.to_scipy_sparse_array(g, format="csr")
        ranks = rng.random(n)
        probs = np.r_[0, np.sort(ranks), 1]
        got = measure(scan, adj.indptr.astype(np.int32), adj.indices.astype(np.int32), ranks, probs)
        for point, pair in zip(probs, got):
            active = g.subgraph(np.flatnonzero(ranks < point))
            sizes = sorted(map(len, nx.connected_components(active)), reverse=True)
            assert np.array_equal(pair, (sizes + [0, 0])[:2])


def simulate():
    scan = compile_scan()
    check_scan(scan)
    shape = (len(DEGREES), len(SIZES), GRAPH_COUNT, len(PROBS), 4)
    sums = np.zeros(shape, dtype=np.int32)
    for ki, degree in enumerate(DEGREES):
        for mi, n in enumerate(SIZES):
            for rep in range(GRAPH_COUNT):
                gs, ss, ds = np.random.SeedSequence([SEED, degree, n, rep]).spawn(3)
                static = np.random.default_rng(ss)
                dynamic = np.random.default_rng(ds)
                g = nx.fast_gnp_random_graph(n, degree/(n-1), seed=int(gs.generate_state(1)[0]))
                adj = nx.to_scipy_sparse_array(g, format="csr", dtype=np.int8)
                offsets, neighbors = adj.indptr.astype(np.int32), adj.indices.astype(np.int32)
                # At probability x, a vertex is online exactly when its rank < x.
                # Zero ranks encode the all-online initial condition across the grid.
                ranks = np.zeros(n)
                for sample in range(SNAPSHOTS):
                    sums[ki, mi, rep, :, :2] += measure(scan, offsets, neighbors, static.random(n))
                    # Exact CTMC kernel over 10 or 3 relaxation times: copy the old
                    # state with probability exp(-dt/tau), otherwise refresh at x.
                    refresh = dynamic.random(n) >= np.exp(-10 if sample == 0 else -3)
                    ranks[refresh] = dynamic.random(refresh.sum())
                    sums[ki, mi, rep, :, 2:] += measure(scan, offsets, neighbors, ranks)
            print(f"k={degree}, M={n}, {GRAPH_COUNT} paired graphs complete", flush=True)
    np.savez_compressed(ROOT / "paired_runs.npz", component_sums=sums, probabilities=PROBS,
                        sizes=SIZES, degrees=DEGREES, snapshots=SNAPSHOTS, seed=SEED, mu=MU)


if __name__ == "__main__":
    simulate()
