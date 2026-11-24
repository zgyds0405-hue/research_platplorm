"""Vector figures and 600 dpi renderings of the hypergraph experiments."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

from simulate import DEGREES, ORDERS, P_CLOCK, ROOT, MU
from evaluate import giant

COLORS = ("#0072B2","#D55E00","#008768","#A74E87")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,
                     "axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
                     "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":.65,
                     "pdf.fonttype":42,"svg.fonttype":"none","legend.frameon":False})


def data(name):
    return np.genfromtxt(ROOT/name,delimiter=",",names=True,encoding="utf-8-sig")


def canvas(rows,cols,title,height,width=9):
    f,axs=plt.subplots(rows,cols,squeeze=False,figsize=(width,height))
    f.subplots_adjust(left=.095,right=.975,bottom=.57/height,top=1-.97/height,hspace=.56,wspace=.28)
    f.text(.095,1-.12/height,title,weight="bold",va="top",fontsize=13)
    return f,axs


def order_legend(f, extras=()):
    handles=[Line2D([],[],color=c,ls="",marker="s",markerfacecolor="none",label=f"h = {h}")
             for h,c in zip(ORDERS,COLORS)]
    f.legend(handles=handles+list(extras),ncol=len(handles)+len(extras),loc="upper left",
             bbox_to_anchor=(.086,1-.40/f.get_size_inches()[1]),fontsize=8)


def save(fig,name):
    out=ROOT/"figures"
    out.mkdir(exist_ok=True)
    for ext in ("svg","pdf","png"):
        file = out / f"{name}.{ext}"
        fig.savefig(file, dpi=600)
        if ext == "svg":
            lines = file.read_text(encoding="utf-8").splitlines()
            file.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


def phase():
    raw=np.load(ROOT/"observations.npz")
    probs=raw["p"]
    f,axs=canvas(3,2,"Steady-state connectivity of uniform hypergraphs",7.7)
    order_legend(f,[Line2D([],[],ls="--",color="#555555",label="Theoretical threshold")])
    display=np.searchsorted(probs,np.round(np.linspace(.02,.98,25),3))
    for zi,z in enumerate(DEGREES):
        for i,title in enumerate(("Static site percolation","Dynamic activation")):
            ax=axs[zi,i]
            thresholds=[(z*(h-1))**(-1/(h-1)) for h in ORDERS]
            for hi,(h,c,pc) in enumerate(zip(ORDERS,COLORS,thresholds)):
                v=raw["sums"][hi,zi,1,:,i,:,0].mean(axis=0)/int(raw["snapshots"])/4000
                ax.scatter(probs[display],v[display],marker="s",s=16,facecolors="none",edgecolors=c,lw=.86,zorder=2)
                same=np.flatnonzero(np.isclose(thresholds,pc))
                pos=np.flatnonzero(same==hi).item()
                ax.axvline(pc,pos/len(same),(pos+1)/len(same),color=c,ls=(0,(4,3)),lw=.82)
            ax.set(title=f"{title}  ·  z = {z}",xlim=(0,1),ylim=(-.012,1.01))
            if i==0: ax.set_ylabel("Largest component, $L_1/N$")
            if zi==2: ax.set_xlabel("Retention probability, $p$" if i==0 else "Stationary online probability, $p_*$")
    save(f,"phase")


def thresholds():
    rows=data("thresholds.csv")
    f,axs=canvas(1,3,"Extrapolated percolation thresholds",3.65,9.8)
    for z,ax in zip(DEGREES,axs[0]):
        v=rows[rows["z"]==z]
        for j,(kind,c,mark) in enumerate([("static","#0072B2","s"),("dynamic","#D55E00","o")]):
            xx=np.arange(4)+(j-.5)*.24
            ax.vlines(xx,v[kind+"_lo"],v[kind+"_hi"],colors=c,lw=1.4)
            ax.scatter(xx,v[kind+"_fit"],s=24,color=c,marker=mark)
        ax.scatter(np.arange(4),v["pc_theory"],c="#282828",marker="_",s=60,linewidths=1.4)
        ax.set(xticks=np.arange(4),xticklabels=ORDERS,xlabel="Hyperedge size, $h$",title=f"Mean hyperdegree z = {z}",ylim=(.07,.71))
        ax.yaxis.set_major_locator(MultipleLocator(.1))
    axs[0,0].set_ylabel("Critical node probability")
    f.legend(handles=[Line2D([],[],marker=m,ls="",color=c,label=t) for m,c,t in
             [("s","#0072B2","Static"),("o","#D55E00","Dynamic"),("_","#282828","Theory")]],
             loc="upper left",bbox_to_anchor=(.086,.90),ncol=3,fontsize=8)
    save(f,"thresholds")


def scaling():
    peaks = data("finite_peaks.csv")
    fits = data("thresholds.csv")
    fig, axes = canvas(2, 2, "Finite-size scaling at mean hyperdegree z = 4", 6.2)
    order_legend(fig, [Line2D([], [], ls="", marker="x", color="#333333", label="Theory at infinite size")])
    for column, group in enumerate(("static", "dynamic")):
        peak_ax, size_ax = axes[:, column]
        peak_ax.set_title("Static site percolation" if column == 0 else "Dynamic activation")
        for h, color in zip(ORDERS, COLORS):
            rows = peaks[(peaks["h"] == h) & (peaks["z"] == 4)]
            row = fits[(fits["h"] == h) & (fits["z"] == 4)][0]
            n = rows["N"]
            x = n**(-1/3)
            estimates = rows[group + "_peak"]
            peak_ax.vlines(x, rows[group + "_lo"], rows[group + "_hi"], color=color, lw=.85)
            peak_ax.scatter(x, estimates, marker="s", s=19, edgecolors=color, facecolors="white", zorder=3)
            slope = np.polyfit(x, estimates, 1)[0]
            xx = np.linspace(0, x.max(), 100)
            peak_ax.plot(xx, row[group + "_fit"] + slope*xx, color=color, lw=.9)
            peak_ax.scatter(0, row["pc_theory"], marker="x", color=color, s=24, clip_on=False)
            sizes = rows["S_" + group + "_at_pc"] * n
            size_ax.scatter(n, sizes, marker="s", s=19, edgecolors=color, facecolors="white")
            power = row[group + "_L1_exponent"]
            c = np.exp(np.mean(np.log(sizes) - power*np.log(n)))
            size_ax.plot(n, c*n**power, color=color, lw=.9)
        peak_ax.set(xlabel=r"$N^{-1/3}$", xlim=(-.003,.086), ylim=(.22,.64))
        size_ax.set(xscale="log", yscale="log", xlabel="Number of nodes, $N$", ylim=(35,570))
        size_ax.set_xticks(n, labels=[f"{int(v/1000)}k" for v in n])
        size_ax.minorticks_off()
    axes[0,0].set_ylabel(r"Peak of mean $L_2$")
    axes[1,0].set_ylabel(r"Mean $L_1$ at theoretical $p_c$")
    save(fig, "scaling")


def clocks():
    rows=data("clock.csv")
    f,axs=canvas(3,2,"Relaxation and persistence in physical time",7.7)
    order_legend(f,[Line2D([],[],lw=1,color="#333333",label="Theory")])
    t=np.linspace(0,400,401)
    for ci,speed in enumerate((1,2)):
        for hi,(h,c) in enumerate(zip(ORDERS,COLORS)):
            v=rows[(rows["h"]==h)&(rows["speed"]==speed)]
            decay=np.exp(-t*speed*MU/(1-P_CLOCK))
            q=P_CLOCK+(1-P_CLOCK)*decay
            theories=[q**h,np.array([giant(x,h,4) for x in q]),(q**h-P_CLOCK**h)/(1-P_CLOCK**h)]
            for j,(field,theo) in enumerate(zip(("active_edges","S","edge_corr"),theories)):
                ax=axs[j,ci]
                ax.plot(t,theo,color=c,lw=.95)
                ax.scatter(v["time"],v[field],marker="s",s=21,facecolors="white",edgecolors=c,lw=.95)
        for j,name in enumerate(("Active hyperedge fraction","Largest component, $L_1/N$","Hyperedge autocorrelation")):
            ax=axs[j,ci]
            ax.set(xlim=(-4,405),ylim=(-.035,1.04))
            if ci==0: ax.set_ylabel(name)
            if j==0: ax.set_title(f"Switching rates ×{speed}")
            if j==2: ax.set_xlabel("Time lag")
            else: ax.set_xlabel("Time from all-online start")
    save(f,"clocks")


def events():
    rows=data("events.csv")
    f,axs=canvas(1,2,"What changes when one node comes online?",3.95)
    ax,bx=axs[0]
    xs=np.arange(4)
    for i,(h,c) in enumerate(zip(ORDERS,COLORS)):
        pc=(4*(h-1))**(-1/(h-1))
        r=rows[(rows["h"]==h)&np.isclose(rows["p"],pc)][0]
        ax.errorbar(i,r["restored_edges"],yerr=r["restored_half"],fmt="s",color=c,
                    mfc="none",capsize=3)
        ax.scatter(i,r["theory_restored"],marker="_",color="#252525",s=120)
        bx.errorbar(i-.09,r["P_r_ge2"],yerr=r["r2_half"],fmt="s",color=c,mfc=c,capsize=3)
        bx.errorbar(i+.09,r["P_r_ge3"],yerr=r["r3_half"],fmt="s",color=c,mfc="white",capsize=3)
    for a in axs[0]:
        a.set(xticks=xs,xticklabels=ORDERS,xlabel="Hyperedge size, $h$")
    ax.set(ylabel="Mean restored hyperedges",title="At each order's theoretical threshold")
    bx.set(ylabel="Probability per activation",ylim=(0,.49),title="Distinct pre-existing components joined")
    ax.legend(handles=[Line2D([],[],color="#333333",marker="_",ls="",label="Theory")])
    bx.legend(handles=[Line2D([],[],marker="s",ls="",color="#333333",mfc=m,label=l)
                       for m,l in [("#333333","At least two"),("white","At least three")]],fontsize=8)
    order_legend(f)
    save(f,"events")


if __name__=="__main__":
    phase()
    thresholds()
    scaling()
    clocks()
    events()
