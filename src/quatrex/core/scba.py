import numpy.linalg as npla

from quatrex.core.config import QuatrexConfig
from quatrex.electron.sigma_phonon import SigmaPhonon
from quatrex.electron.electron_solver import ElectronSolver

from qttools.datastructures.coogroup import COOGroup


class SCBA:
    """Computes the self-consistent Born approximation to convergence.

    Parameters
    ----------
    config_file : Path
        The configuration for the LDOS calculation.

    """

    def __init__(self, config: QuatrexConfig):
        # self.config = config
        self.electron_solver = ElectronSolver(config)

        self.sigma_lesser = COOGroup(
            self.electron_solver.n_energies_per_rank,
            rows=self.electron_solver.hamiltonian.row,
            cols=self.electron_solver.hamiltonian.col,
            pinned=True,
        )
        self.sigma_greater = COOGroup(
            self.electron_solver.n_energies_per_rank,
            rows=self.electron_solver.hamiltonian.row,
            cols=self.electron_solver.hamiltonian.col,
            pinned=True,
        )
        self.g_lesser = COOGroup(
            self.electron_solver.n_energies_per_rank,
            rows=self.electron_solver.hamiltonian.row,
            cols=self.electron_solver.hamiltonian.col,
            pinned=True,
        )
        self.g_greater = COOGroup(
            self.electron_solver.n_energies_per_rank,
            rows=self.electron_solver.hamiltonian.row,
            cols=self.electron_solver.hamiltonian.col,
            pinned=True,
        )

        if config.scba.phonon:

            self.sigma_phonon = SigmaPhonon(config)

            if self.sigma_phonon.model == "greens-function":
                raise NotImplementedError

            elif self.sigma_phonon.model == "pseudo-scattering":
                ...

        self.mix = config.scba.mixing_factor
        self.max_iterations = config.scba.max_iterations

    def _converged(self, *arg) -> bool:
        # Compute difference between current and preceeding self-energy.
        # sigma_retarded = 0.5 * (sigma_greater - sigma_lesser)
        # prev_sigma_retarded = 0.5 * (
        #     self.electron_solver.sigma_greater - self.electron_solver.sigma_lesser
        # )
        # diff_causal = sigma_retarded - prev_sigma_retarded

        # abs_norm_diff_causal = npla.norm(diff_causal)
        # rel_norm_diff_causal = abs_norm_diff_causal / npla.norm(sigma_retarded)
        return False

    def run(self) -> None:

        sigma_lesser = COOGroup(
            self.electron_solver.n_energies_per_rank,
            rows=self.electron_solver.hamiltonian.row,
            cols=self.electron_solver.hamiltonian.col,
        )
        sigma_greater = COOGroup(
            self.electron_solver.n_energies_per_rank,
            rows=self.electron_solver.hamiltonian.row,
            cols=self.electron_solver.hamiltonian.col,
        )

        for __ in range(self.max_iterations):
            # Reset the self-energy.
            self.g_lesser.data[:], self.g_greater.data[:] = (
                self.electron_solver.solve_lesser_greater(
                    self.sigma_lesser, self.sigma_greater
                )
            )

            sigma_lesser.data[:] = 0.0
            sigma_greater.data[:] = 0.0
            if hasattr(self, "sigma_phonon"):
                if self.sigma_phonon.model == "greens-function":
                    raise NotImplementedError

                elif self.sigma_phonon.model == "pseudo-scattering":
                    sigma_phonon_lesser, sigma_phonon_greater = (
                        self.sigma_phonon.pseudo_scattering(
                            self.g_lesser, self.g_greater
                        )
                    )

                sigma_lesser.data[:] += sigma_phonon_lesser.data
                sigma_greater.data[:] += sigma_phonon_greater.data

            # Update self-energy.
            self.sigma_lesser[:] = (
                self.mix * self.sigma_lesser + (1 - self.mix) * sigma_lesser
            )
            self.sigma_greater[:] = (
                self.mix * self.sigma_greater + (1 - self.mix) * sigma_greater
            )

            if self._converged():
                print(f"SCBA converged after {__} iterations.")
                break

        else:  # Did not break, i.e. max_iterations reached.
            print(f"SCBA did not converge after {__} iterations.")
