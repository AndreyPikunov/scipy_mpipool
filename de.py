from mpipool import MPIPool
from mpi4py import MPI

import numpy as np
from scipy.optimize import rosen, differential_evolution

ndim = 2
bounds = np.vstack([np.full(ndim, -5),
                    np.full(ndim,  5)]).T

def func(x):
    rank = MPI.COMM_WORLD.Get_rank()
    loss = np.sum(x**2)
    print(rank, x, loss)
    return loss

with MPIPool() as pool:
    pool.workers_exit()
    result = differential_evolution(func,
                                    bounds,
                                    maxiter=10, popsize=1,
                                    polish=False,
                                    disp=True,
                                    workers=pool.map,
                                    updating='deferred', seed=42)
