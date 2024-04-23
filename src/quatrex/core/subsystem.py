from abc import ABC, abstractmethod
from functools import partial

from mpi4py.MPI import COMM_WORLD

import numpy as np
from quatrex.core.config import OBCConfig, QuatrexConfig
from qttools.obc import sancho_rubio

from qttools.greens_function import inv
from qttools.datastructures.coogroup import COOGroup


class SubsystemSolver(ABC):
    """Abstract core class for subsystem solvers.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object.

    """

    @property
    @abstractmethod
    def system(self) -> str: ...

    def __init__(self, config: QuatrexConfig, *args: tuple, **kwargs: dict) -> None:
        """Initializes the solver."""

        self.energies = np.load(config.input_dir / "energies.npy")

        self.obc = self._configure_obc(getattr(config, self.system).obc)
        self.invert = self._configure_solver_retarded(
            getattr(config, self.system).solver
        )
        self.solver = self._configure_solver_lesser_greater(
            getattr(config, self.system).solver
        )

        (self.n_energies_per_rank, self.energy_slice) = self._compute_energy_slice()

    def _compute_energy_slice(self):
        """Computes the energy slice."""
        n_energies = len(self.energies) // COMM_WORLD.size

        if len(self.energies) % COMM_WORLD.size != 0:
            raise ValueError(
                "The number of energies must be divisible by the number of MPI processes."
            )

        start_energy = n_energies * COMM_WORLD.rank
        end_energy = start_energy + n_energies

        return (n_energies, slice(start_energy, end_energy))

    def _configure_obc(self, obc_config: OBCConfig) -> callable:
        """Configures the OBC algorithm from the config.

        Parameters
        ----------
        obc_config : QuatrexConfig
            The OBC configuration.

        Returns
        -------
        obc : callable
            The configured OBC algorithm.

        """
        if obc_config.algorithm == "sancho-rubio":

            def obc(a_ii, a_ij, a_ji, side: str) -> np.ndarray:
                """Calculates the surface Green's function."""
                return partial(
                    sancho_rubio,
                    max_iterations=obc_config.max_iterations,
                    max_delta=obc_config.max_delta,
                )(a_ii, a_ij, a_ji)

            return obc

        raise NotImplementedError(
            f"OBC algorithm {obc_config.algorithm} not implemented."
        )

    def _configure_solver_retarded(self, solver: str) -> callable:
        """Configures the inversion algorithm from the config.

        Parameters
        ----------
        solver : str
            The inversion algorithm.

        Returns
        -------
        invert : callable
            The configured inversion algorithm.

        """
        if solver == "rgf":
            raise NotImplementedError

        if solver == "inv":
            return inv.inv_retarded

        raise NotImplementedError(f"Solver '{solver}' not implemented.")

    def _configure_solver_lesser_greater(self, solver: str) -> callable:
        """Configures the solver algorithm from the config.

        Parameters
        ----------
        solver : str
            The solver algorithm.

        Returns
        -------
        solver : callable
            The configured solver algorithm.

        """
        if solver == "rgf":
            raise NotImplementedError

        if solver == "inv":
            return inv.inv_lesser_greater

        raise NotImplementedError(f"Solver '{solver}' not implemented.")

    @abstractmethod
    def apply_obc(self, *args, **kwargs) -> tuple:
        """Applies the OBC."""
        ...

    @abstractmethod
    def assemble_system_matrix(self, energy: float) -> COOGroup:
        """Assembles the system matrix for a given energy."""
        ...

    @abstractmethod
    def solve_retarded(self) -> tuple[np.ndarray, np.ndarray]:
        """Solves for the retarded Green's function for all energies."""
        ...

    @abstractmethod
    def solve_lesser_greater(self) -> tuple[np.ndarray, np.ndarray]:
        """Solves for the lesser/greater Green's function for all energies."""
        ...
