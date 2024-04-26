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
- `solve(sigma_lesser, sigma_greater)`
- `assemble_system()`

## `SelfEnergy`
- `__call__(self, g_lesser, g_greater, **kwargs)`

Example:
```
compute_sigma_phonon = PseudoScatteringSigmaPhonon(config.phonon)
compute_sigma_phonon(g_lesser, g_greater, out=sigma_phonon)

compute_sigma_phonon = NEGFSigmaPhonon(config.phonon)
compute_sigma_phonon(g_lesser, g_greater, d_lesser, d_greater, out=sigma_phonon)
```


# Dataclasses

### `Observables`
### `SCBAData`

# Config
- 

