# Copyright 2023-2024 ETH Zurich and QuaTrEx authors. All rights reserved.

from pathlib import Path

from pydantic import BaseModel


class SCBAConfig(BaseModel):
    max_iterations: int = 1000
    tolerance: float = 1e-8

    mixing_factor: float = 0.1

    electron: bool = True
    phonon: bool = False


class StructureConfig(BaseModel):
    temperature: float = 300.0  # K
    num_cells: int


class OBCConfig(BaseModel):
    algorithm: str = "sancho-rubio"

    # Parameters for iterative solvers.
    max_iterations: int = 5000
    max_delta: float = 1e-8


class ElectronConfig(BaseModel):
    eta: float = 1e-6  # eV

    # intrinsic_fermi_level: float = 0.0  # eV
    left_fermi_level: float = 0.0  # eV
    right_fermi_level: float = 0.0  # eV

    solver: str = "inv"

    obc: OBCConfig = OBCConfig()


class PhononConfig(BaseModel):
    energy_range: list[float]
    energy_step: float = 1e-3  # eV

    solver: str = "inv"

    # "greens-function" or "pseudo-scattering"
    model: str = "pseudo-scattering"

    deformation_potential: float = 25e-3  # eV
    energy: float = 50e-3  # eV

    obc: OBCConfig = OBCConfig()


class QuatrexConfig(BaseModel):
    structure: StructureConfig

    electron: ElectronConfig
    phonon: PhononConfig

    scba: SCBAConfig

    # --- Directory paths ----------------------------------------------
    simulation_dir: Path = Path("./quatrex/")

    @property
    def input_dir(self) -> Path:
        return self.simulation_dir / "inputs/"
