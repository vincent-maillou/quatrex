# Classes
## `SCSP`
- `PoissonSolver`
- `SCBA`
- `Observables`

- `__init__(self, config)`
- `run()`

## `PoissonSolver`
- `QuatrexConfig`
- `solve()`

## `SCBA`
- `QuatrexConfig`
- `ElectronSolver`
- `Observables`
- `SCBAData`

- `CoulombScreeningSolver = None`
- `PhotonSolver = None`
- `PhononSolver = None`

- `CoulombScreeningPolarization = None`
- `PhotonPolarization = None` 
- `PhononPolarization = None`

- `CoulombScreeningSelfEnergy = None`
- `PhotonSelfEnergy = None` 
- `PhononSelfEnergy = None`

- `__init__(self, config)`

- `_compute_screened_coulomb_interaction()`
- `_compute_photon_interaction()`
- `_compute_phonon_interaction()`
- `run()`

## `SubsystemSolver`
- `__init__(self, config)`
- `solve(sigma_lesser, sigma_greater)`
- `_assemble_system_matrix()`
- `_apply_obc()`

## `SelfEnergy`
- `__init__(self, config)`
- `__call__(self, g_lesser, g_greater, **kwargs)`

Example:
```python
compute_sigma_phonon = PseudoScatteringSigmaPhonon(config.phonon)
compute_sigma_phonon(g_lesser, g_greater, out=sigma_phonon)

compute_sigma_phonon = NEGFSigmaPhonon(config.phonon)
compute_sigma_phonon(g_lesser, g_greater, d_lesser, d_greater, out=sigma_phonon)
```


# Dataclasses

### `Observables`
###### Electron
- `electron_ldos`
- `electron_density`
- `hole_density`
- `electron_current`

- `electron_electron_scattering_rate`
- `electron_photon_scattering_rate`
- `electron_phonon_scattering_rate`

- `sigma_retarded_density`
- `sigma_lesser_density`
- `sigma_greater_density`

###### Coulomb Screening
- `w_retarded_density`
- `w_lesser_density`
- `w_greater_density`

- `p_retarded_density`
- `p_lesser_density`
- `p_greater_density`

###### Photon
- `pi_photon_retarded`
- `pi_photon_lesser`
- `pi_photon_greater`

- `d_photon_retarded`
- `d_photon_lesser`
- `d_photon_greater`

- `photon_current`

###### Phonon
- `pi_phonon_retarded`
- `pi_phonon_lesser`
- `pi_phonon_greater`
- `d_phonon_retarded`
- `d_phonon_lesser`
- `d_phonon_greater`

- `thermal_current`

### `SCBAData`
- `sigma_retarded`
- `sigma_lesser`
- `sigma_greater`
- `g_retarded`
- `g_lesser`
- `g_greater`

- `p_retarded`
- `p_lesser`
- `p_greater`
- `w_retarded`
- `w_lesser`
- `w_greater`

- `pi_photon_retarded`
- `pi_photon_lesser`
- `pi_photon_greater`
- `d_photon_retarded`
- `d_photon_lesser`
- `d_photon_greater`

- `pi_phonon_retarded`
- `pi_phonon_lesser`
- `pi_phonon_greater`
- `d_phonon_retarded`
- `d_phonon_lesser`
- `d_phonon_greater`


# Config

Tree:
- `QuatrexConfig`
	- `SCSPConfig`
	- `SCBAConfig`
	- `PoissonConfig`
	- `ElectronConfig`
		- `OBCConfig`
	- `CoulombScreeningConfig`
		- `OBCConfig` 
	- `PhotonConfig`
		- `OBCConfig`
	- `PhononConfig`
		- `OBCConfig`


Components:
- `QuatrexConfig`
	- `scsp`: `SCSPConfig`
	- `scba`: `SCBAConfig`
	- `poisson`: `PoissonConfig`
	- `electron`: `ElectronConfig`
	- `coulomb_screening`: `CoulombScreeningConfig`
	- `photon`: `PhotonConfig`
	- `phonon`: `PhononConfig`

- `SCSPConfig`
	- `min_iterations`
	- `max_iterations`
	- `convergence_tol`
	- `mixing_factor`

- `SCBAConfig`
	- `min_iterations`
	- `max_iterations`
	- `convergence_tol`
	- `mixing_factor`
	- `observables_interval`
	- `coulomb_screening: bool = False`
	- `photon: bool = False`
	- `phonon: bool = False`
	- `interaction_range`
	- `coulomb_screening_interaction_range: float`
	- `photon_interaction_range: float`
	- `phonon_interaction_range: float`

- `PoissonConfig`
	- `model: str = "point-charge"`
	- `mixing_factor`
	- `convergence_tol`
	- `max_iterations`

- `OBCConfig`
	- `algorithm: str = "sancho-rubio"`
	- `max_iterations: int = 5000`
	- `convergence_tol: float = 1e-8`
	- `lyapunov_method`: `str`

- `ElectronConfig`
	- `solver`: `str`
	- `obc`: `OBCConfig`

	- `eta`

	- `fermi_level`: 
	- `left_fermi_level`: 
	- `right_fermi_level`

	- `temperature`:
	- `left_temperature`:
	- `right_temperature`

- `CoulombScreeningConfig`
	- `solver`
	- `obc`: `OBCConfig`

- `PhotonConfig`
	- `solver`
	- `obc`: `OBCConfig`

	- `model`:  `pseudo-scattering`
	- `photon_energy`
	- `photon_density`
	- `polarization`

- `PhononConfig`
	- `solver`
	- `obc`: `OBCConfig`

	- `model`:  `pseudo-scattering`
	- `phonon_energy`
	- `deformation_potential`
