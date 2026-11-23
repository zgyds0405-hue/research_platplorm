from pathlib import Path
import numpy as np
import networkx as nx
base=Path(__file__).parent
rng=np.random.default_rng(2026112308)
N=2000
# Save the first network's 50 snapshots and the mean over all 20 runs.
samples=np.zeros((4,3,3,50,71))
means=np.zeros((4,3,3,71))
for family in range(2):
    for s,param in enumerate(([5,10,20],[20,40,60])[family]):
        for run in range(20):
            seed=int(rng.integers(2**31))
            G=nx.barabasi_albert_graph(N,param,seed=seed) if family==0 else nx.watts_strogatz_graph(N,param,.2,seed=seed)
            adj=nx.to_scipy_sparse_array(G,format='csr',dtype=np.int32)
            for i,l in enumerate([.010,.020]):
                row=family*2+i
                for j,m in enumerate([.005,.010,.015]):
                    p=l/(l+m)
                    persistence=np.exp(-(l+m)*200)
                    # Exact transition from t=0 (all online) to t=10000.
                    online=rng.random(N)<p+(1-p)*np.exp(-10000*(l+m))
                    for t in range(50):
                        online=rng.random(N)<p+(online-p)*persistence
                        degrees=(adj@online.astype(np.int32))[online]
                        freq=np.bincount(degrees,minlength=71)[:71]/online.sum()
                        if run==0: samples[row,j,s,t]=freq
                        means[row,j,s]+=freq/1000
np.savez(base/'eight.npz',samples=samples,means=means)
