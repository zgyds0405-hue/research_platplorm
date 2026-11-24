"""Direct stationary sampling of the state immediately before a node comes online."""
import csv

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from simulate import ORDERS, REPS, ROOT, SEED, graph

N, Z, EVENTS = 4000, 4, 200


def one_event(edges, pivot_edges, online, pivot, verify=False):
    online[pivot]=False
    active=online[edges].all(axis=1)
    # Intact hyperedges can each be connected as a star for component queries.
    # An incomplete edge contributes no part of its star.
    intact=edges[active]
    left=np.repeat(intact[:,0],edges.shape[1]-1)
    right=intact[:,1:].ravel()
    adj=csr_matrix((np.ones(len(left),np.int8),(left,right)),shape=(N,N))
    _,labels=connected_components(adj,directed=False)
    sizes=np.bincount(labels[online],minlength=labels.max()+1)
    largest=int(sizes.max())
    neighbors=edges[pivot_edges[pivot]]
    restored=neighbors[(online[neighbors].sum(axis=1)==edges.shape[1]-1)]
    linked=np.unique(labels[restored[restored!=pivot]])
    merged=1+int(sizes[linked].sum())
    gain=max(0,merged-largest)
    if verify:
        online[pivot]=True
        new=edges[online[edges].all(axis=1)]
        i=np.repeat(new[:,0],edges.shape[1]-1)
        j=new[:,1:].ravel()
        _,post=connected_components(csr_matrix((np.ones(len(i)),(i,j)),shape=(N,N)),directed=False)
        after=np.bincount(post[online])
        assert after.max()-largest==gain and np.count_nonzero(after)-np.count_nonzero(sizes)==1-len(linked)
    return len(restored),len(linked),gain


def run():
    with (ROOT/"event_samples.csv").open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(["h","z","N","graph_id","p","sample","restored_edges","merged_components","L1_gain"])
        for h in ORDERS:
            pc=(Z*(h-1))**(-1/(h-1))
            for rep in range(REPS):
                gs=np.random.SeedSequence([SEED,h,Z,N,rep]).spawn(1)[0]
                edges=graph(N,h,Z,np.random.default_rng(gs))
                rng=np.random.default_rng([SEED,911,h,rep])
                flat=edges.ravel()
                counts=np.bincount(flat,minlength=N)
                incident=np.split(np.argsort(flat)//h,np.cumsum(counts)[:-1])
                for p in sorted({.8*pc,pc,1.2*pc,.75}):
                    for sample in range(EVENTS):
                        pivot=rng.integers(N)
                        result=one_event(edges,incident,rng.random(N)<p,pivot,sample==0)
                        w.writerow([h,Z,N,rep+1,p,sample+1,*result])
            print(f"event sampling complete for h={h}",flush=True)


if __name__=="__main__":
    run()
