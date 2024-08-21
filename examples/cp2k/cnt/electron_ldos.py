import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quatrex.core.quatrex_config import parse_config
from quatrex.post_processing.plot_ldos import get_averages, spectral_plot

path = os.path.dirname(__file__)
output_dir = Path(f"{path}/outputs")
input_dir = Path(f"{path}/inputs")
num_cells = len(np.load(input_dir / "block_sizes.npy"))

energies = np.load(input_dir / "electron_energies.npy")
ldos = np.load(output_dir / "electron_ldos.npy", allow_pickle=True)
# potential = np.load(input_dir / "potential.npy")

config = parse_config(f"{path}/config.toml")

left_fermi_level = config.electron.left_fermi_level
right_fermi_level = config.electron.right_fermi_level


fig, ax_ldos = plt.subplots(figsize=(8, 5))

num_energies = len(energies)
average_ldos = get_averages(ldos, num_energies, num_cells, method="rolling")

spectral_plot(ax_ldos, average_ldos, energies, cmap="viridis", colorbar_label="LDOS")

plt.show()
