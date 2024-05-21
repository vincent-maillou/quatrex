# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from qttools.datastructures import DSBSparse

from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy


class SigmaPhonon(ScatteringSelfEnergy):
    """Computes the lesser electron-photon self-energy."""

    def __init__(
        self,
        config: QuatrexConfig,
        g_lesser: DSBSparse,
        g_greater: DSBSparse,
    ) -> None:
        """Initializes the self-energy."""
        self.model = config.phonon.model

        self.g_lesser = g_lesser
        self.g_greater = g_greater

    def compute(self) -> np.ndarray:
        """Computes the electron-photon self-energy."""
        if self.model == "negf":
            raise NotImplementedError
        elif self.model == "deformation-potential":
            return self._quasi()
