from pathlib import Path
from fig5 import plot_figure5
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
from scipy.stats import binom
from PIL import Image
import json
base=Path(__file__).parent
out=base/'figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'dejavuserif','font.size':9,
    'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':9,'axes.linewidth':.65,
    'xtick.labelsize':8,'ytick.labelsize':8,'xtick.major.width':.55,'ytick.major.width':.55,
    'svg.fonttype':'none','pdf.fonttype':42,'axes.axisbelow':True,'savefig.facecolor':'white'})
rgb=['#ff0000','#008000','#0000ff']; marks=['o','^','s']
rates=[.005,.010,.015]; sizes=[2000,4000,6000]
def canvas(rows,title,height):
    fig,axes=plt.subplots(rows,3,squeeze=False,figsize=(11.1,height))
    fig.suptitle(title,x=.077,y=1-.16/height,ha='left',va='top',fontsize=16)
    fig.subplots_adjust(left=.079,right=.983,top=1-1.13/height,bottom=.59/height,wspace=.29,hspace=.59)
    for ax in axes.flat:
        ax.grid(linestyle='--',linewidth=.48,color='#b8b8b8')
        for spine in ax.spines.values(): spine.set_color('#676767')
    return fig,axes
def head(ax,index,title):
    ax.set_title('('+chr(97+index)+')  '+title,loc='left',pad=7)
def legend(fig,handles,columns):
    h=fig.get_size_inches()[1]
    fig.legend(handles=handles,ncol=columns,loc='upper left',bbox_to_anchor=(.074,1-.53/h),
        borderaxespad=0,frameon=False,handlelength=2.5,columnspacing=1.8)
def save(fig,n):
    for ext in ['svg','pdf','png']:
        fig.savefig(out/f'Fig{n}.{ext}',dpi=600)
    plt.close(fig)
def key(color,label,marker='',ls='-',fill=True):
    return Line2D([],[],color=color,label=label,marker=marker,linestyle=ls,linewidth=1.0,
                  markersize=4.5,markerfacecolor=color if fill else 'white',markeredgewidth=.7)

plot_figure5(out)

fig,axes=canvas(1,'Stationary distribution of online network size',3.87)
legend(fig,[key(c,rf'$\mu={m:.3f}$',z,'',False) for c,m,z in zip(rgb,rates,marks)]+[key('#555555','Binomial theory')],4)
occ=np.load(base/'six.npy'); n=np.arange(1001)
tv=[]
for i,l in enumerate(rates):
    ax=axes[0,i]
    for j,(c,m,z) in enumerate(zip(rgb,rates,marks)):
        theo=binom.pmf(n,1000,l/(l+m)); mask=theo>1e-6
        ax.plot(n[mask],theo[mask],c=c,lw=.85)
        f=occ[i,j]; mask=f>1.e-6
        ax.plot(n[mask],f[mask],ls='',marker=z,mec=c,mfc='none',mew=.40,ms=2.45)
        tv.append(.5*abs(f-theo).sum())
    ax.set(xlim=(0,1000),ylim=(0,.035),xlabel=r'$N$')
    ax.xaxis.set_major_locator(MultipleLocator(200)); ax.yaxis.set_major_locator(MultipleLocator(.005))
    if i==0: ax.set_ylabel(r'$P(N=n)$')
    head(ax,i,rf'$\lambda={l:.3f}$')
save(fig,6)

fig,axes=canvas(1,'Mean online network size versus hidden rate',3.87)
legend(fig,[key(c,rf'$N_0={N}$',z) for c,N,z in zip(rgb,sizes,marks)]+[key('#d1a500','Theory (dashed)',ls='--')],4)
means=np.load(base/'seven.npy').mean(axis=-1); mus=np.arange(1,61)*.0005; mae=[]
for i,l in enumerate(rates):
    ax=axes[0,i]
    for j,(c,N,z) in enumerate(zip(rgb,sizes,marks)):
        ax.plot(mus,means[i,j],c=c,lw=.7,marker=z,ms=3.2,mec='#242424',mew=.36)
        smooth=np.linspace(mus[0],mus[-1],400)
        ax.plot(smooth,N*l/(l+smooth),color='#ffc600',ls=(0,(3,2)),lw=.9)
        mae.append(abs(means[i,j]-N*l/(l+mus)).mean())
    ax.set(xlim=(0,.03),ylim=(0,6000),xlabel=r'$\mu$')
    ax.set_xticks([0,.005,.01,.015,.02,.025,.03])
    ax.set_xticklabels(['0','.005','.010','.015','.020','.025','.030'])
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    if i==0: ax.set_ylabel(r'$\mathrm{E}[N]$')
    head(ax,i,rf'$\lambda={l:.3f}$')
save(fig,7)

fig,axes=canvas(4,'Degree distributions of online subnetworks',11.2)
data=np.load(base/'eight.npz'); samples=data['samples']; means=data['means']
colors=[rgb[0],rgb[2],rgb[1]]
legend(fig,[key(c,rf'$m={a}$ (SF) / $K={b}$ (SW)',mark,'') for c,a,b,mark in zip(colors,[5,10,20],[20,40,60],marks)],3)
for row in range(4):
    family=['SF','SF','SW','SW'][row]
    l=[.01,.02][row%2]
    for j,m in enumerate(rates):
        ax=axes[row,j]
        for s,(c,z) in enumerate(zip(colors,marks)):
            f=samples[row,j,s]
            t,k=np.nonzero(f)
            ax.scatter(k,f[t,k],s=2.6,marker=z,c=c,linewidths=0,alpha=.29)
            ax.plot(np.arange(71),means[row,j,s],lw=.62,color=c)
        ax.set(xlim=(0,70),ylim=(0,.3 if row<2 else .2),xlabel=r'$k$')
        ax.xaxis.set_major_locator(MultipleLocator(10)); ax.yaxis.set_major_locator(MultipleLocator(.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if j==0: ax.set_ylabel(r'$P(k)$')
        head(ax,row*3+j,rf'{family}, $\lambda={l:.3f},\ \mu={m:.3f}$')
save(fig,8)
print('TV range',min(tv),max(tv),'MAE range',min(mae),max(mae))
