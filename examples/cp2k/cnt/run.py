import os
import time

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from quatrex.core.quatrex_config import parse_config
from quatrex.core.scba import SCBA

example_path = os.path.dirname(__file__)

if __name__ == "__main__":
    config = parse_config(f"{example_path}/config.toml")
    print("The work directory is:", example_path)

    scba = SCBA(config)

    tic = time.perf_counter()
    scba.run()
    toc = time.perf_counter()

    print(f"Leaving SCBA after: {(toc - tic):.2f} s")
    if comm.rank == 0:
        np.save(
            f"{example_path}/outputs/electron_ldos.npy",
            scba.observables.electron_ldos,
        )
        np.save(
            f"{example_path}/outputs/electron_density.npy",
            scba.observables.electron_density,
        )
        np.save(
            f"{example_path}/outputs/hole_density.npy",
            scba.observables.hole_density,
        )
