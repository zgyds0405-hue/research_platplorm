"""Independent connectivity, paired-data, plotting and table checks."""
import csv
import gzip

import numpy as np
from openpyxl import load_workbook
from PIL import Image

from simulate import (DEGREES, ORDERS, REPS, ROOT, SEED, SIZES, SAMPLES,
                      evolve_ranks, graph, library, measure)


def bfs(edges, mask):
    # Reference implementation builds a bipartite incidence graph of intact edges only.
    active = np.flatnonzero(mask)
    intact = edges[mask[edges].all(axis=1)]
    n = len(mask)
    adj = [[] for _ in range(n + len(intact))]
    for j,members in enumerate(intact,n):
        for v in members:
            adj[j].append(v)
            adj[v].append(j)
    seen = set()
    sizes = []
    for start in active:
        if start in seen: continue
        seen.add(start)
        stack=[start]; count=0
        while stack:
            v=stack.pop()
            count+=v<n
            for w in adj[v]:
                if w not in seen:
                    stack.append(w)
                    seen.add(w)
        sizes.append(count)
    first,second=(sorted(sizes,reverse=True)+[0,0])[:2]
    return [first,second,len(intact),len(active),len(sizes)]


def run():
    fn=library()
    rng=np.random.default_rng(6391)
    for h in ORDERS:
        e=graph(40,h,4,rng)
        for ranks in (rng.random(40),np.zeros(40),np.ones(40)):
            ps=np.r_[0,.25,.5,.75,1.]
            truth=[bfs(e,ranks<p) for p in ps]
            np.testing.assert_array_equal(measure(fn,e,ranks,ps),truth)
    # Three online nodes cannot communicate via a four-member edge with one offline.
    e=np.array([[0,1,2,3]],dtype=np.int32)
    assert measure(fn,e,np.array([0.,0.,0.,1.]),np.array([.5]))[0].tolist()==[1,1,0,3,3]
    db=np.load(ROOT/"observations.npz")
    probs,sums=db["p"],db["sums"]
    assert sums.shape==(4,3,5,10,2,len(probs),5)
    for hi,zi,ni,r in [(0,0,0,2),(2,1,1,1),(3,2,4,0)]:
        h,z,n=ORDERS[hi],DEGREES[zi],SIZES[ni]
        gr,st,dy,_=map(np.random.default_rng,np.random.SeedSequence([SEED,h,z,n,r]).spawn(4))
        e=graph(n,h,z,gr)
        ranks=np.zeros(n); sub=np.array([.25,.50,.75])
        checked=np.zeros((2,len(sub),5))
        for s in range(SAMPLES):
            evolve_ranks(ranks,10 if s==0 else 3,dy)
            static=st.random(n)
            for j,p in enumerate(sub):
                checked[0,j]+=bfs(e,static<p);checked[1,j]+=bfs(e,ranks<p)
        np.testing.assert_array_equal(checked,sums[hi,zi,ni,r][:,np.searchsorted(probs,sub)])
    read=lambda name: np.genfromtxt(ROOT/name,names=True,delimiter=",",encoding="utf-8-sig")
    peaks=read("finite_peaks.csv"); fits=read("thresholds.csv")
    assert len(peaks)==60 and len(fits)==12
    for hi,h in enumerate(ORDERS):
        for zi,z in enumerate(DEGREES):
            g=peaks[(peaks["h"]==h)&(peaks["z"]==z)]
            peak=probs[sums[hi,zi,:,:,:,:,1].mean(axis=1).argmax(axis=-1)]
            np.testing.assert_array_equal(peak,np.column_stack((g["static_peak"],g["dynamic_peak"])))
            b=np.polyfit(np.array(SIZES)**(-1/3),peak,1)[1]
            row=fits[(fits["h"]==h)&(fits["z"]==z)][0]
            np.testing.assert_allclose(b,[row["static_fit"],row["dynamic_fit"]])
    with gzip.open(ROOT/"full_scan.csv.gz","rt") as f:
        count=0
        for row in csv.DictReader(f):
            s,d,err=map(float,(row["static_S"],row["dynamic_S"],row["relative_error_pct"]))
            assert np.isclose(err,100*abs(d-s)/s,atol=1e-10)
            count+=1
        assert count==60*len(probs)
    events=read("events.csv")
    assert sum(events["event_count"])==32000 and np.allclose(events["P_r0"]+events["P_r1"]+events["P_r_ge2"],1)
    wb=load_workbook(ROOT/"hypergraph_results.xlsx",read_only=True,data_only=True)
    assert wb["全部稳态结果"].max_row==count+1
    for stem in ("phase","thresholds","scaling","clocks","events"):
        with Image.open(ROOT/"figures"/(stem+".png")) as im: assert abs(im.info['dpi'][0]-600)<.1
        assert "<image " not in (ROOT/"figures"/(stem+".svg")).read_text()
    print("Reference BFS, 60 peak pairs, 12 intercept pairs, 32,000 event records, 59,280 errors and exports checked.")


if __name__=="__main__":
    run()
