# Copyright 2023-2024 ETH Zurich and QuaTrEx authors. All rights reserved.
import numpy as np

from scipy import sparse

from quatrex.core.config import QuatrexConfig
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.statistics import fermi_dirac

from qttools.datastructures.coogroup import COOGroup


class ElectronSolver(SubsystemSolver):
    """Solves for the lesser electron Green's function."""

    system = "electron"

    def __init__(self, config: QuatrexConfig, **kwargs) -> None:
        """Initializes the solver."""
        super().__init__(config)

        self.hamiltonian: sparse.coo_array = sparse.load_npz(
            config.input_dir / "hamiltonian.npz"
        ).tocoo()
        potential = np.load(config.input_dir / "potential.npy")
        self.potential = sparse.diags(potential, format="coo")

        self.eta = config.electron.eta

    def apply_obc(
        self,
        system_matrices: COOGroup,
        return_sigma_obc: bool = False,
        occupancies_l: tuple[float, float] = None,
        occupancies_g: tuple[float, float] = None,
    ) -> tuple[COOBatch, COOBatch | None, COOBatch | None]:
        """Applies the OBC to the system matrix."""

        m_01 = system_matrices.get_block(0, 1)
        m_10 = system_matrices.get_block(1, 0)
        m_nm = system_matrices.get_block(-1, -2)
        m_mn = system_matrices.get_block(-2, -1)

        m_00 = system_matrices.get_block(0, 0)
        m_nn = system_matrices.get_block(-1, -1)

        m_00_temp = np.array([m + 1j * self.eta * np.eye(*m.shape) for m in m_00])
        m_nn_temp = np.array([m + 1j * self.eta * np.eye(*m.shape) for m in m_nn])

        g_00 = self.obc(m_00_temp, m_01, m_10, side="left")
        g_nn = self.obc(m_nn_temp, m_nm, m_mn, side="right")

        sigma_obc_00 = m_10 @ g_00 @ m_01
        sigma_obc_nn = m_mn @ g_nn @ m_nm

        system_matrices.set_block(0, 0, m_00 - sigma_obc_00)
        system_matrices.set_block(-1, -1, m_nn - sigma_obc_nn)

        if not return_sigma_obc:
            return system_matrices

        if occupancies_l is None or occupancies_g is None:
            raise ValueError("Occupancies must be set to compute sigma_lesser.")

        sigma_l_obc_00 = m_10 @ (occupancies_l[0] * (g_00.conj().T - g_00)) @ m_01
        sigma_l_obc_nn = m_mn @ (occupancies_l[-1] * (g_nn.conj().T - g_nn)) @ m_nm

        sigma_g_obc_00 = -m_10 @ (occupancies_g[0] * (g_00.conj().T - g_00)) @ m_01
        sigma_g_obc_nn = -m_mn @ (occupancies_g[-1] * (g_nn.conj().T - g_nn)) @ m_nm

        sigma_l_obc = COOGroup(
            self.n_energies_per_rank,
            rows=self.hamiltonian.row,
            cols=self.hamiltonian.col,
        )
        sigma_g_obc = COOGroup(
            self.n_energies_per_rank,
            rows=self.hamiltonian.row,
            cols=self.hamiltonian.col,
        )

        sigma_l_obc.set_block(0, 0, sigma_l_obc_00)
        sigma_l_obc.set_block(-1, -1, sigma_l_obc_nn)

        sigma_g_obc.set_block(0, 0, sigma_g_obc_00)
        sigma_g_obc.set_block(-1, -1, sigma_g_obc_nn)

        return system_matrices, sigma_l_obc, sigma_g_obc

    def assemble_system_matrix(self, energy: float) -> bsp.BSparse:
        """Assembles the system matrix for a given energy.

        Parameters
        ----------
        energy : float
            The energy.

        Returns
        -------
        system_matrix : bsp.BSparse
            The system matrix.

        """
        energy_diag = bsp.BCOO.from_array(
            energy * np.eye(self.hamiltonian.shape[0], dtype=np.complex128),
            sizes=(self.hamiltonian.row_sizes, self.hamiltonian.col_sizes),
        )
        if self.sigma_lesser is None or self.sigma_greater is None:
            return energy_diag - self.hamiltonian - self.potential

        index = np.where(np.isclose(self.energies, energy))[0][0]

        sigma_lesser = self.sigma_lesser[index]
        sigma_greater = self.sigma_greater[index]

        sigma_causal = 0.5 * (sigma_greater - sigma_lesser)

        return energy_diag - self.hamiltonian - self.potential - sigma_causal
