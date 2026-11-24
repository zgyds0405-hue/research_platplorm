"""Place critical-region diagnostics beneath the original static / Markov panels."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

from analyze import BETA_WINDOWS, theory
from run_critical import ROOT, WINDOW

sys.path.insert(0, str(ROOT.parent))
from percolation import draw as draw_overview

COLORS = ("#0072B2", "#D55E00", "#009E73")
GROUPS = ("static", "dynamic")


def choose(x, targets):
    return np.unique(np.abs(x[:,None]-targets).argmin(axis=0))


def points(ax, x, y, color, marker="s", hollow=True):
    return ax.plot(x, y, marker=marker, ls="", ms=4.1, mew=.85, color=color,
                   mfc="white" if hollow else color)[0]


def export(fig, name):
    for ext in ("pdf", "svg", "png"):
        file = ROOT / f"{name}.{ext}"
        fig.savefig(file, dpi=600)
        if ext == "svg":
            file.write_text("\n".join(line.rstrip() for line in file.read_text().splitlines())+"\n")
    plt.close(fig)


def overview_figure(scan):
    prior = np.loadtxt(ROOT.parent / "summary.csv", skiprows=1, delimiter=",")
    fig, axes = plt.subplots(4, 2, figsize=(11.6, 13.4), layout="constrained")
    fig.get_layout_engine().set(w_pad=.12, h_pad=.10, hspace=.06, wspace=.05)
    eps_grid = np.r_[np.linspace(-.24, 0, 40), np.geomspace(.001,.30,180)]
    exact = np.array([theory(e) for e in eps_grid])
    main_lo, main_hi = BETA_WINDOWS[0]
    beta = np.genfromtxt(ROOT / "beta_fits.csv", names=True, delimiter=",", dtype=None,
                         encoding="utf-8")
    degrees, sizes = np.unique(scan["k"]), np.unique(scan["N"])
    for col, group in enumerate(GROUPS):
        title = "Static site percolation" if col == 0 else r"Markov activation ($\mu=0.01$)"
        symbol = "p" if col == 0 else "p_*"
        y_name, ci_name = group+"_S", group+"_95_half"
        ax, zoom, power, critical = axes[:,col]
        draw_overview(ax, prior, col)
        ax.legend(loc="upper left", fontsize=8, labelspacing=.5, frameon=True,
                  facecolor="white", edgecolor="none", framealpha=1)
        ax.set_title(f"{'ab'[col]}   {title}", loc="left", weight="bold")
        zoom.plot(eps_grid, exact, color="#353535", lw=1.15)
        zoom.axvline(0, color="#aaaaaa", lw=.8, ls="--")
        power.axvspan(main_lo, main_hi, color="#eaf1f6", lw=0)
        pos = eps_grid >= .022
        power.plot(eps_grid[pos], exact[pos], color="#353535", lw=1.2, label="Exact ER limit")
        reference = np.array([.023, .25])
        power.plot(reference, 2*reference, color="#555555", ls="--", lw=.95, label=r"$2\epsilon$ ($\beta=1$)")
        for k,color in zip(degrees,COLORS):
            data = scan[scan["k"]==k]
            for n, marker, hollow in ((4000, "s", True),(sizes[-1], "o", False)):
                series = data[data["N"]==n]
                ix = choose(series["epsilon"], np.linspace(-.22,.22,16))
                points(zoom, series["epsilon"][ix], k*series[y_name][ix],color,marker,hollow)
            series = data[data["N"]==sizes[-1]]
            dx = series["epsilon"]
            ix = choose(dx, np.geomspace(.026,.25,15))
            power.errorbar(dx[ix],k*series[y_name][ix],yerr=k*series[ci_name][ix],fmt="s",
                           color=color, ms=4.1, mew=.8, mfc="white", lw=.7, capsize=1.5)
            at = data[data["epsilon"]==0]
            critical.errorbar(at["N"],at[y_name],yerr=at[ci_name],fmt="s",color=color,
                              ms=4.4,mfc="white",mew=.9,lw=.8,capsize=2)
            ref = at[y_name][2]*(sizes/sizes[2])**(-1/3)
            critical.plot(sizes,ref,color=color,lw=.75,ls="--")
        zoom_legend = [Line2D([],[],color=c,ls="",marker="s",mfc="white",label=f"k = {k:g}") for k,c in zip(degrees,COLORS)]
        zoom_legend += [
            Line2D([],[],color="#353535",ls="",marker="s",mfc="white",label="M = 4,000"),
            Line2D([],[],color="#353535",ls="",marker="o",label=f"M = {sizes[-1]:,.0f}"),
            Line2D([],[],color="#353535",lw=1.15,label="Exact ER limit")]
        zoom.legend(handles=zoom_legend,fontsize=7.2,loc="upper left",ncol=2,
                    handletextpad=.4,columnspacing=.7,labelspacing=.4)
        zoom.set(xlim=(-.235,.235),ylim=(-.006,.475),
                 xlabel=rf"Reduced occupation, $\epsilon=k{symbol}-1$",ylabel=r"Rescaled order parameter, $kS$")
        zoom.set_title(f"{'cd'[col]}   Critical window",loc="left",weight="bold")
        power.set(xscale="log",yscale="log",xlim=(.022,.277),ylim=(.038,.56),
                  xlabel=r"Distance above threshold, $\epsilon$",ylabel=r"Rescaled order parameter, $kS$")
        power.set_title(f"{'ef'[col]}   Supercritical scaling  ·  M = {sizes[-1]:,.0f}",loc="left",weight="bold")
        power.set_xticks([.025,.05,.10,.20],labels=["0.025","0.05","0.10","0.20"])
        power.set_yticks([.05,.10,.20,.40],labels=["0.05","0.10","0.20","0.40"])
        power.legend(loc="upper left",fontsize=7.5)
        group_name = "Static" if col == 0 else "Markov"
        main = beta[(beta["epsilon_min"] == main_lo) & (beta["N"] == sizes[-1])
                    & (beta["fit"] == "First-order correction") & (beta["group"] == group_name)]
        power.text(.97, .205, "Curvature-corrected fits", ha="right", va="bottom",
                   transform=power.transAxes, fontsize=7.1, color="#454545")
        for pos, (row, color) in enumerate(zip(main, COLORS)):
            power.text(.97, .15-.055*pos,
                       rf"$k={row['k']}\,:\,\hat{{\beta}}={row['beta']:.3f}$",
                       ha="right", va="bottom", color=color, transform=power.transAxes, fontsize=7.5)
        critical.set(xscale="log",yscale="log",xlim=(820,1.22*sizes[-1]),ylim=(.0019,.10),
                     xlabel="Number of nodes, $M$",ylabel=r"Order parameter at threshold, $S(p_c)$")
        critical.set_title(f"{'gh'[col]}   Finite-size scaling at the threshold",loc="left",weight="bold")
        critical.set_xticks(sizes,labels=[f"{n/1000:g}k" for n in sizes])
        critical.set_yticks([.0025,.005,.01,.025,.05,.10],labels=["0.0025","0.005","0.01","0.025","0.05","0.10"])
        critical.legend(handles=[Line2D([],[],ls="--",color="#555555",label=r"$M^{-1/3}$ guide")],
                        loc="upper right",fontsize=7.5)
        for panel in (power,critical):
            panel.xaxis.set_minor_locator(NullLocator())
            panel.yaxis.set_minor_locator(NullLocator())
    export(fig,"er_near_critical")


def collapse_figure(scan):
    fig, axes = plt.subplots(2,3,figsize=(11,6.5),layout="constrained")
    scales = ("#606060","#352A87","#0072B2","#008768","#BA5C00","#A74683")
    markers = ("o","s","^","D","v","P")
    sizes = np.unique(scan["N"])
    for row, group in enumerate(GROUPS):
        for col,k in enumerate(np.unique(scan["k"])):
            ax = axes[row,col]
            for n,c,m in zip(sizes,scales,markers):
                data = scan[(scan["N"]==n)&(scan["k"]==k)]
                ix = choose(data["scaled_x"],WINDOW[::2])
                y = k * np.cbrt(n/k) * data[group+"_S"][ix]
                points(ax,data["scaled_x"][ix],y,c,m,n!=sizes[-1])
            x = np.linspace(1.5,4,50)
            ax.plot(x,2*x,color="#777777",lw=.85,ls="--")
            ax.axvline(0,color="#b0b0b0",lw=.65,ls=":")
            ax.set(xlim=(-4.25,4.25),ylim=(0,8.85),title=f"{'Static' if row==0 else 'Markov'}  ·  k = {k:g}")
            if col==0:
                ax.set_ylabel(r"$kS(M/k)^{1/3}$")
            if row==1:
                ax.set_xlabel(r"$\epsilon(M/k)^{1/3}$")
    handles = [Line2D([],[],ls="",marker=m,color=c,mfc="white",label=f"M = {n:,.0f}")
               for n,c,m in zip(sizes,scales,markers)]
    handles += [Line2D([],[],color="#777777",ls="--",label=r"$2x$ guide")]
    axes[0,0].legend(handles=handles,loc="upper left",fontsize=6.9,ncol=2,labelspacing=.35)
    export(fig,"er_scaling_collapse")


if __name__ == "__main__":
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.linewidth":.7,"axes.titlesize":10,
                         "legend.frameon":False,"svg.fonttype":"none","pdf.fonttype":42})
    scan = np.genfromtxt(ROOT / "near_critical.csv", names=True, delimiter=",")
    overview_figure(scan)
    collapse_figure(scan)
