"""Transient network size from two fixed initial online counts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def simulate_counts(rng, total, initial, online_rate, hidden_rate, times):
    """Sample 20 independent trajectories exactly at the observation times."""
    states = np.full(20, initial)
    counts = np.empty((len(times), len(states)), dtype=int)
    rate = online_rate + hidden_rate
    stationary = online_rate / rate
    for index, elapsed in enumerate(np.diff(times, prepend=0.0)):
        switched = -np.expm1(-rate * elapsed)
        departed = rng.binomial(states, (1 - stationary) * switched)
        arrived = rng.binomial(total - states, stationary * switched)
        states = states - departed + arrived
        counts[index] = states
    return counts


def plot_figure5(output):
    output.mkdir(exist_ok=True, parents=True)
    times = np.geomspace(0.1, 10000, 2100)
    sizes = (2000, 4000, 6000)
    colors = ("#0072B2", "#D55E00", "#009E73")
    theory = "#343434"
    with plt.rc_context({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 7, "axes.linewidth": 0.55, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "axes.axisbelow": True,
    }):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65), facecolor="white")
        fig.subplots_adjust(left=.092, right=.985, top=.755, bottom=.115, wspace=.25, hspace=.45)
        fig.text(.092, .965, "Online network size over time", weight="bold", fontsize=12, va="top")
        low_color = "#CC79A7"
        low_theory = "#572749"
        scenarios = [(n, n, c) for n, c in zip(sizes, colors)]
        scenarios.append((4000, 400, low_color))
        handles = [Line2D([], [], color=c, lw=1.6, label=f"{initial:,} / {n:,} online")
                   for n, initial, c in scenarios]
        handles.extend([
            Line2D([], [], color=theory, lw=.8, ls="--", label="Mean, full start"),
            Line2D([], [], color=low_theory, lw=.8, ls=":", label="Mean, 10% start"),
        ])
        order = (0, 3, 1, 4, 2, 5)
        fig.legend(handles=[handles[i] for i in order], loc="upper left", bbox_to_anchor=(.086, .903),
                   ncol=3, frameon=False, borderaxespad=0, handlelength=2.8, columnspacing=1.8,
                   fontsize=7)
        full_rng = np.random.default_rng(2026112305)
        low_rng = np.random.default_rng(2026112306)
        for row, online_rate in enumerate((.010, .020)):
            for col, hidden_rate in enumerate((.005, .010, .015)):
                ax = axes[row, col]
                rate = online_rate + hidden_rate
                p = online_rate / rate
                for total, initial, color in scenarios:
                    full = initial == total
                    rng = full_rng if full else low_rng
                    counts = simulate_counts(rng, total, initial, online_rate, hidden_rate, times)
                    mean = total * p + (initial - total * p) * np.exp(-rate * times)
                    ax.plot(times, counts[:, 0], color=color, lw=.8)
                    ax.plot(times, mean, color=theory if full else low_theory,
                            lw=.55, ls=(0, (3, 4)) if full else ":")
                label = chr(ord("a") + 3 * row + col)
                ax.set_title(rf"$\mathbf{{{label}}}$   $\lambda={online_rate:.3f},\ \mu={hidden_rate:.3f}$",
                             fontsize=7.2, loc="left", pad=4)
                ax.set(xscale="log", xlim=(.1, 10000), ylim=(0, 6300),
                       yticks=[0, 2000, 4000, 6000], xticks=[.1, 1, 10, 100, 1000, 10000])
                ax.minorticks_off()
                ax.tick_params(width=.5, length=2, pad=2, labelleft=(col == 0), labelbottom=(row == 1))
                ax.grid(axis="y", color="#dadada", lw=.4)
                if col == 0:
                    ax.set_ylabel("Online nodes", fontsize=7)
                if row == 1 and col == 1:
                    ax.set_xlabel("Time, $t$", fontsize=7)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(output / f"Fig5.{ext}", dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    plot_figure5(Path(__file__).parent / "figures")
