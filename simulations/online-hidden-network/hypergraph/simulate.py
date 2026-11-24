"""Static and dynamic site percolation on simple random uniform hypergraphs."""
import ctypes
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np

ROOT = Path(__file__).resolve().parent
SIZES = (2000, 4000, 8000, 16000, 32000)
ORDERS = (2, 3, 4, 6)
DEGREES = (2, 4, 8)
REPS, SAMPLES, SEED, MU = 10, 8, 492173, .01
CRITICAL = [(z*(h-1))**(-1/(h-1)) for h in ORDERS for z in DEGREES]
PROBS = np.unique(np.r_[np.arange(10, 991) / 1000, CRITICAL])
LAGS = np.array([0., 5., 10., 25., 50., 100., 200., 400.])
CLOCKS = (1, 2)
P_CLOCK = .65


def graph(n, h, z, rng):
    """Choose m distinct uniform h-subsets without replacement."""
    m = round(n*z/h)
    pool = np.empty((0, h), dtype=np.int32)
    while len(pool) < m:
        count = int((m-len(pool))*1.02) + 32
        rows = np.sort(rng.integers(n, size=(count,h), dtype=np.int32), axis=1)
        valid = (np.diff(rows, axis=1) != 0).all(axis=1)
        pool = np.unique(np.vstack((pool, rows[valid])), axis=0)
    return np.ascontiguousarray(pool[rng.choice(len(pool), m, replace=False)])


def library():
    with TemporaryDirectory() as folder:
        lib = str(Path(folder)/"scan.so")
        subprocess.run(["g++","-O3","-std=c++17","-shared","-fPIC",str(ROOT/"scan.cpp"),"-o",lib],check=True)
        fn = ctypes.CDLL(lib).scan
    arr = np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
    fn.argtypes = [ctypes.c_int,ctypes.c_int,arr,arr,arr,ctypes.c_int,arr,arr,arr]
    fn.restype = None
    return fn


def measure(fn, edges, ranks, probs):
    # A hyperedge becomes active at the largest of its members' uniform ranks.
    edge_rank = ranks[edges].max(axis=1)
    vertices = np.argsort(ranks).astype(np.int32)
    groups = np.argsort(edge_rank).astype(np.int32)
    vc = np.searchsorted(ranks[vertices], probs).astype(np.int32)
    ec = np.searchsorted(edge_rank[groups], probs).astype(np.int32)
    out = np.empty((len(probs),5),np.int32)
    fn(len(ranks),edges.shape[1],edges,vertices,groups,len(probs),vc,ec,out)
    return out


def evolve_ranks(ranks, elapsed, rng):
    refresh = rng.random(len(ranks)) >= np.exp(-elapsed)
    ranks[refresh] = rng.random(refresh.sum())


def clock_experiment(fn, edges, n, rng):
    """All-online relaxation and independently initialized stationary joint observations."""
    h = edges.shape[1]
    p = P_CLOCK
    out = np.zeros((len(CLOCKS),len(LAGS),5))
    for c, multiplier in enumerate(CLOCKS):
        rate = multiplier * MU / (1-p)
        for _ in range(SAMPLES):
            start = np.zeros(n)
            stationary = rng.random(n)
            base_v = stationary < p
            base_e = base_v[edges].all(axis=1)
            for j, dt in enumerate(np.diff(LAGS,prepend=0)):
                if dt:
                    evolve_ranks(start,rate*dt,rng)
                    evolve_ranks(stationary,rate*dt,rng)
                obs = measure(fn,edges,start,np.array([p]))[0]
                v = stationary < p
                e = v[edges].all(axis=1)
                # Joint-moment covariance estimator using known stationary mean.
                cv = np.mean((base_v-p)*(v-p))/(p*(1-p))
                pe = p**h
                ce = np.mean((base_e-pe)*(e-pe))/(pe*(1-pe))
                out[c,j] += [obs[0]/n,obs[2]/len(edges),obs[3]/n,cv,ce]
    return out / SAMPLES


def run():
    fn = library()
    totals=np.empty((len(ORDERS),len(DEGREES),len(SIZES),REPS,2,len(PROBS),5),np.int32)
    clocks=np.empty((len(ORDERS),REPS,len(CLOCKS),len(LAGS),5))
    for hi,h in enumerate(ORDERS):
        for zi,z in enumerate(DEGREES):
            for ni,n in enumerate(SIZES):
                for r in range(REPS):
                    seeds=np.random.SeedSequence([SEED,h,z,n,r]).spawn(4)
                    gr,st,dy,clock=map(np.random.default_rng,seeds)
                    edges=graph(n,h,z,gr)
                    ranks=np.zeros(n)
                    sample_sum=np.zeros((2,len(PROBS),5),np.int32)
                    for s in range(SAMPLES):
                        evolve_ranks(ranks,10 if s==0 else 3,dy)
                        sample_sum[0]+=measure(fn,edges,st.random(n),PROBS)
                        sample_sum[1]+=measure(fn,edges,ranks,PROBS)
                    totals[hi,zi,ni,r]=sample_sum
                    if n==4000 and z==4:
                        clocks[hi,r]=clock_experiment(fn,edges,n,clock)
            print(f"completed h={h}, z={z}",flush=True)
    np.savez_compressed(ROOT/"observations.npz",sums=totals,clock_means=clocks,orders=ORDERS,
                        degrees=DEGREES,sizes=SIZES,p=PROBS,lags=LAGS,replicates=REPS,
                        snapshots=SAMPLES,seed=SEED,mu=MU,clock_p=P_CLOCK,clock_multipliers=CLOCKS)


if __name__=="__main__":
    run()
