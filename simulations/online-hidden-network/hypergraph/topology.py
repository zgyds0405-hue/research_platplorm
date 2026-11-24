"""Check hyperdegree and incidence-graph four-cycles on the simulated topologies."""
from itertools import combinations
import numpy as np

from simulate import DEGREES, ORDERS, REPS, ROOT, SEED, SIZES, graph
from evaluate import write


def run():
    rows=[]
    for h in ORDERS:
        for z in DEGREES:
            for n in SIZES:
                for rep in range(REPS):
                    seed=np.random.SeedSequence([SEED,h,z,n,rep]).spawn(1)[0]
                    edges=graph(n,h,z,np.random.default_rng(seed))
                    deg=np.bincount(edges.ravel(),minlength=n)
                    codes=np.concatenate([edges[:,i].astype(np.int64)*n+edges[:,j] for i,j in combinations(range(h),2)])
                    pairs,freq=np.unique(codes,return_counts=True)
                    shared=pairs[freq>1]
                    nodes=np.unique(np.r_[shared//n,shared%n]).size
                    cycles=(freq*(freq-1)//2).sum()
                    rows.append((h,z,n,rep+1,len(edges),deg.mean(),sum(deg*(deg-1))/deg.sum(),cycles,nodes/n))
    write("topology.csv","h z N graph_id edge_count mean_hyperdegree excess_hyperdegree incidence_four_cycles four_cycle_vertex_fraction",rows)


if __name__=="__main__":
    run()
