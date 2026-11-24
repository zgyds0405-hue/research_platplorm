"""Compute graph-paired uncertainties, finite-size fits, clock and event benchmarks."""
import csv
import gzip

import numpy as np
from scipy.optimize import brentq
from scipy.stats import t

from simulate import CLOCKS, DEGREES, LAGS, MU, ORDERS, P_CLOCK, REPS, ROOT, SAMPLES, SIZES

B = 2500
T95 = t.ppf(.975, REPS-1)


def giant(p,h,z):
    """Conditional giant fraction a solves a = 1-exp(-z*p^(h-1)*(1-(1-a)^(h-1)))."""
    branch = z*(h-1)*p**(h-1)
    if branch <= 1+1e-12:
        return 0.
    def residual(a):
        if a == 1:
            return -np.expm1(-z*p**(h-1))-1
        return -np.expm1(z*p**(h-1)*np.expm1((h-1)*np.log1p(-a)))-a
    return p*brentq(residual,1e-14,1.)


def limits(x):
    return tuple(np.quantile(x,[.025,.975]))


def paired_stats(x):
    return x.mean(axis=0), T95*x.std(axis=0,ddof=1)/np.sqrt(len(x))


def write(name, cols, rows):
    path=ROOT/name
    stream=gzip.open(path,"wt",encoding="utf-8",newline="") if name.endswith(".gz") else path.open("w",encoding="utf-8-sig",newline="")
    with stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(cols.split())
        writer.writerows(rows)


def analyze():
    data=np.load(ROOT/"observations.npz")
    probs=data["p"]
    assert np.all(np.diff(probs)>0)
    sums=data["sums"]
    rng=np.random.default_rng(241253)
    sizes=np.array(SIZES,dtype=float)
    x=sizes**(-1/3)
    fit=np.linalg.pinv(np.column_stack([np.ones(len(sizes)),x]))[0]
    trim=np.linalg.pinv(np.column_stack([np.ones(len(sizes)-1),x[1:]]))[0]
    slope=np.linalg.pinv(np.column_stack([np.ones(len(sizes)),np.log(sizes)]))[1]
    curves=[]; peaks=[]; fits=[]
    for hi,h in enumerate(ORDERS):
        for zi,z in enumerate(DEGREES):
            pc=(z*(h-1))**(-1/(h-1))
            at=np.flatnonzero(probs==pc).item()
            phase=np.array([giant(p,h,z) for p in probs])
            peak=[]; boots=[]; crit=[]
            for ni,n in enumerate(SIZES):
                obs=sums[hi,zi,ni]/SAMPLES
                mean=obs.mean(axis=0)
                l1=obs[:,:,:,0]/n
                sl,dl=l1[:,0],l1[:,1]
                s,sm=paired_stats(sl); d,dm=paired_stats(dl)
                diff,delta=paired_stats(dl-sl)
                assert np.all(s>0)
                median_edges=round(n*z/h)
                for j,p in enumerate(probs):
                    curves.append((h,z,n,p,MU*p/(1-p),*([s[j],sm[j],d[j],dm[j]]),
                                   abs(diff[j])/s[j]*100,diff[j]-delta[j],diff[j]+delta[j],
                                   phase[j],mean[0,j,1],mean[1,j,1],mean[0,j,2]/median_edges,
                                   mean[1,j,2]/median_edges,p**h,mean[1,j,3]/n,mean[1,j,4]))
                l2=obs[:,:,:,1]
                pair=probs[l2.mean(axis=0).argmax(axis=1)]
                weights=rng.multinomial(REPS,np.ones(REPS)/REPS,size=B)
                boot=np.column_stack([probs[(weights @ l2[:,kind]).argmax(axis=1)] for kind in (0,1)])
                peak.append(pair); boots.append(boot)
                crit.append(mean[:,at,0])
                peaks.append((h,z,n,pc,pair[0],*limits(boot[:,0]),pair[1],*limits(boot[:,1]),
                              pair[1]-pair[0],*limits(boot[:,1]-boot[:,0]),s[at],d[at],abs(d[at]-s[at])/s[at]*100))
            peak=np.array(peak)
            intercept=fit@peak
            bi=np.einsum("n,nba->ba",fit,boots)
            fits.append((h,z,pc,intercept[0],*limits(bi[:,0]),intercept[1],*limits(bi[:,1]),
                         intercept[1]-intercept[0],*limits(bi[:,1]-bi[:,0]),*(trim@peak[1:]),
                         *(slope@np.log(crit))))
    cols="h z N p lambda static_S static_CI_half dynamic_S dynamic_CI_half relative_error_pct diff_lo diff_hi theory_S static_L2 dynamic_L2 static_edge_fraction dynamic_edge_fraction theory_edge_fraction dynamic_online_fraction dynamic_component_count"
    write("full_scan.csv.gz",cols,curves)
    write("steady_selected.csv",cols,[r for r in curves if r[2]==4000 and (r[3] in (.1,.25,.4,.5,.6,.65,.75,.9) or r[3]==(r[1]*(r[0]-1))**(-1/(r[0]-1)))])
    write("finite_peaks.csv","h z N pc_theory static_peak static_lo static_hi dynamic_peak dynamic_lo dynamic_hi diff diff_lo diff_hi S_static_at_pc S_dynamic_at_pc relative_error_pct",peaks)
    write("thresholds.csv","h z pc_theory static_fit static_lo static_hi dynamic_fit dynamic_lo dynamic_hi diff diff_lo diff_hi static_trim dynamic_trim static_L1_exponent dynamic_L1_exponent",fits)
    clock=[]
    for hi,h in enumerate(ORDERS):
        for ci,c in enumerate(CLOCKS):
            rate=c*MU/(1-P_CLOCK)
            for j,lag in enumerate(LAGS):
                obs=data['clock_means'][hi,:,ci,j]
                mean,margin=paired_stats(obs)
                decay=np.exp(-rate*lag)
                q=P_CLOCK+(1-P_CLOCK)*decay
                edge_ac=(q**h-P_CLOCK**h)/(1-P_CLOCK**h)
                clock.append((h,c,c*MU*P_CLOCK/(1-P_CLOCK),c*MU,lag,*mean,*margin,
                              giant(q,h,4),q**h,q,decay,edge_ac))
    write("clock.csv","h speed lambda mu time S active_edges online node_corr edge_corr S_half edges_half online_half nodecorr_half edgecorr_half theory_S theory_active theory_online theory_nodecorr theory_edgecorr",clock)
    events=np.loadtxt(ROOT/"event_samples.csv",delimiter=",",skiprows=1)
    rows=[]
    for h,p in np.unique(events[:,[0,4]],axis=0):
        selected=events[(events[:,0]==h)&(events[:,4]==p)]
        graph_means=[]
        for r in range(REPS):
            a=selected[selected[:,3]==r+1]
            restored,merge,gain=a[:,6:].T
            graph_means.append([restored.mean(),(merge==0).mean(),(merge==1).mean(),
                                (merge>=2).mean(),(merge>=3).mean(),gain.mean(),
                                (gain>0).mean(),(gain>=10).mean(),(gain>=100).mean()])
        mean,margin=paired_stats(np.array(graph_means))
        rows.append((int(h),p,len(selected),*mean,*margin,4*p**(h-1),int((selected[:,-1]>=100).sum())))
    write("events.csv","h p event_count restored_edges P_r0 P_r1 P_r_ge2 P_r_ge3 mean_gain P_gain_positive P_gain_ge10 P_gain_ge100 restored_half r0_half r1_half r2_half r3_half gain_half pos_half ge10_half ge100_half theory_restored count_gain_ge100",rows)


if __name__=="__main__":
    analyze()
