# QuaTrEx
Quantum Transport at the Exascale and Beyond

## How to install
```
1. Create a conda environment
    $ conda env create -f environment.yml

2. Install Qttools
    $ cd path/to/qttools
    $ pip install --no-dependencies -e .

3. Install QuaTrEx
    $ cd path/to/quatrex
    $ pip install --no-dependencies -e .
```

Note: CuPy and mpi4py are shipped with the environment and configured w.r.t local clusters. If you are using a different cluster, you might need to install them manually.


### Just in case..
```
conda install conda-forge::numpy conda-forge::scipy conda-forge::matplotlib conda-forge::pydantic anaconda::pytest conda-forge::pytest-cov conda-forge::pytest-mpi conda-forge::coverage conda-forge::black conda-forge::isort conda-forge::ruff conda-forge::sphinx conda-forge::sqlite sqlite=3.45.3 conda-forge::pre_commit

conda install -c conda-forge mpi4py mpich=4.2

conda install -c conda-forge cupy cuda-version=11.4
```