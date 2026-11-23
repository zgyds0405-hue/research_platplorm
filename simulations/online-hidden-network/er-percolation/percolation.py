"""Paired ER site-percolation and online/hidden simulations."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
M = 4000
DEGREES = (2, 4, 8)
MU = 0.01
GRAPHS = 24
SAMPLES = 8
SEED = 20261124
# Integer thousandths keep the grid identical, including each theoretical threshold.
grid = {10, 990, *range(25, 1000, 25)}
for degree in DEGREES:
    threshold = 1000 // degree
    grid.update(range(threshold - 30, threshold + 31, 5))
# Visit the original grid first so its seeded samples are exactly reproducible.
original_grid = sorted(grid)
grid.update(range(10, 991, 10))
for degree in DEGREES:
    threshold = 1000 // degree
    grid.update(range(threshold - 10, threshold + 11, 2))
SIMULATION_ORDER = np.array(original_grid + sorted(grid.difference(original_grid))) / 1000
PROBABILITIES = np.sort(SIMULATION_ORDER)
# Approximately even spacing, including the measured endpoints and all three thresholds.
DISPLAY_PROBABILITIES = np.array([
    .01, .06, .10, .125, .18, .22, .25, .30, .34, .38, .42, .46, .50,
    .54, .58, .62, .66, .70, .74, .78, .82, .86, .90, .94, .99,
])


def largest_fraction(adjacency, online):
    if not online.any():
        return 0.
    active = adjacency[online][:, online]
    _, labels = connected_components(active, directed=False)
    return np.bincount(labels).max() / M


def evolve(online, stationary, elapsed_in_relaxation_times, rng):
    # Exact independent two-state transitions over the full elapsed interval.
    persistence = np.exp(-elapsed_in_relaxation_times)
    probability = stationary + (online - stationary) * persistence
    return rng.random(M) < probability


def simulate():
    rows = []
    for k in DEGREES:
        for graph_id in range(GRAPHS):
            graph_stream, static_stream, dynamic_stream = np.random.SeedSequence(
                [SEED, k, graph_id]).spawn(3)
            graph = nx.fast_gnp_random_graph(M, k / (M - 1), seed=int(graph_stream.generate_state(1)[0]))
            adj = nx.to_scipy_sparse_array(graph, format="csr", dtype=np.int8)
            static = np.random.default_rng(static_stream)
            dynamic = np.random.default_rng(dynamic_stream)
            for x in SIMULATION_ORDER:
                activation = MU * x / (1 - x)
                stationary = activation / (activation + MU)
                online = np.ones(M, dtype=bool)
                values = []
                for sample in range(SAMPLES):
                    # First snapshot at 10/(lambda+mu), then every 3/(lambda+mu).
                    online = evolve(online, stationary, 10 if sample == 0 else 3, dynamic)
                    retained = static.random(M) < x
                    values.append((largest_fraction(adj, retained), largest_fraction(adj, online),
                                   retained.mean(), online.mean()))
                rows.append((k, graph_id, x, activation, MU, *np.mean(values, axis=0)))
        print(f"Finished k={k}", flush=True)
    return np.array(rows)


def write_csv(name, header, rows):
    with (ROOT / name).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header.split())
        writer.writerows(rows)


def summarize(rows):
    summary = []
    critical_t = t.ppf(.975, GRAPHS - 1)
    for k in DEGREES:
        for x in PROBABILITIES:
            group = rows[(rows[:, 0] == k) & (rows[:, 2] == x)]
            assert len(group) == GRAPHS
            static, dynamic = group[:, 5], group[:, 6]
            estimates = []
            for series in (static, dynamic, dynamic - static):
                mean = series.mean()
                margin = critical_t * series.std(ddof=1) / np.sqrt(GRAPHS)
                estimates.extend((mean, mean - margin, mean + margin))
            summary.append((k, x, group[0, 3], *estimates))
    return np.array(summary)


def draw(ax, summary, dynamic):
    mean_column = 6 if dynamic else 3
    for k, color in zip(DEGREES, ("#0072B2", "#D55E00", "#009E73")):
        group = summary[(summary[:, 0] == k) & np.isin(summary[:, 1], DISPLAY_PROBABILITIES)]
        threshold = 1.0 / k
        ax.axvline(threshold, color=color, ls=(0, (5, 3)), lw=1.35, zorder=1)
        ax.scatter(group[:, 1], group[:, mean_column], marker="s", s=32,
                   facecolors="none", edgecolors=color, linewidths=1.0, zorder=2,
                   label=rf"$k={k},\quad p_c={threshold:.3f}$")
    ax.set(xlim=(0, 1), ylim=(0, 1),
           xlabel="Stationary online probability, $p_*$" if dynamic else "Retention probability, $p$",
           ylabel="Largest component, $L_1/M$")
    ax.xaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.set_axisbelow(True)
    ax.margins(x=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelspacing=.6)


def export(summary):
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "mathtext.fontset":"dejavusans",
                         "font.size":9, "axes.spines.top":False, "axes.spines.right":False,
                         "axes.linewidth":.65, "svg.fonttype":"none", "pdf.fonttype":42})
    titles = ("Static site percolation", r"Dynamic activation ($\mu=0.01$)")
    for stem, kinds in [("comparison", (0, 1)), ("static", (0,)), ("dynamic", (1,))]:
        width = 14.0 if len(kinds) == 2 else 7.0
        fig, axes = plt.subplots(1, len(kinds), figsize=(width, 4.65), squeeze=False,
                                 layout="constrained")
        for ax, kind in zip(axes.flat, kinds):
            draw(ax, summary, kind)
            label = ("a   ", "b   ")[kind] if len(kinds) == 2 else ""
            ax.set_title(label + titles[kind], loc="left", fontsize=10, weight="bold", pad=9)
        for ext in ("svg", "pdf", "png"):
            fig.savefig(out / f"{stem}.{ext}", dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    results = simulate()
    write_csv("replicates.csv", "k graph_id x lambda mu static_L1 dynamic_L1 static_occupancy dynamic_occupancy", results)
    summary = summarize(results)
    write_csv("summary.csv", "k x lambda static_mean static_low static_high dynamic_mean dynamic_low dynamic_high paired_diff diff_low diff_high", summary)
    export(summary)
