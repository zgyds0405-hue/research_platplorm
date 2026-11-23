import ctypes as c
from pathlib import Path
import numpy as np
base=Path(__file__).parent
sim=c.CDLL(str(base/'events.so')).observe
pointer=c.POINTER(c.c_double)
sim.argtypes=[c.c_int,c.c_int,c.c_double,c.c_double,c.c_double,c.c_uint64,pointer]
sim.restype=c.c_double

def simulate(rng,N,l,m,duration,hist=None):
    p=l/(l+m)
    q=p+(1-p)*np.exp(-(l+m)*10000)
    start=rng.binomial(N,q)
    seed=int(rng.bit_generator.random_raw())
    ptr=None if hist is None else hist.ctypes.data_as(pointer)
    return sim(N,start,l,m,duration,seed,ptr)

rates=[.005,.010,.015]
rng=np.random.default_rng(2026112306)
occ=np.zeros((3,3,1001))
for i,l in enumerate(rates):
    for j,m in enumerate(rates):
        for r in range(20):
            simulate(rng,1000,l,m,10000,occ[i,j])
occ/=20*10000
assert np.allclose(occ.sum(axis=-1),1)
np.save(base/'six.npy',occ)
rng=np.random.default_rng(2026112307)
means=np.zeros((3,3,60,20))
for i,l in enumerate(rates):
    for j,N in enumerate([2000,4000,6000]):
        for k,m in enumerate(np.arange(1,61)*.0005):
            for r in range(20):
                means[i,j,k,r]=simulate(rng,N,l,m,5000)
    print('Completed rate',l,flush=True)
np.save(base/'seven.npy',means)
