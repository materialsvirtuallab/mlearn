# mlearn

NOTE: This package has been deprecated and all code has been moved to the 
updated [maml](https://github.com/materialsvirtuallab/maml) package. Please
use maml from henceforth. This package is retained for reference but it is
archived and will no longer be updated or maintained.

The mlearn package is a benchmark suite for machine learning interatomic 
potentials for materials science. It enables a seamless way to develop 
various potentials and provides LAMMPS-driven properties predictor with 
developed potentials as plugins.

# Installation

The usage of mlearn requires installation of specific packages and the plugins 
in [LAMMPS](https://lammps.sandia.gov/). Please see 
[detailed installation instructions](docs/install.md) for all descriptors.

# Jupyter Notebook Examples

* [Gaussian Approximation Potential (GAP)](notebooks/GAP_example/example.ipynb)
* [Moment Tensor Potential (MTP)](notebooks/MTP_example/example.ipynb)
* [Neural Network Potential (NNP)](notebooks/NNP_example/example.ipynb)
* [Linear Spectral Neighbor Analysis Potential (SNAP)](notebooks/SNAP_example/example.ipynb) 
  and [quadratic SNAP](notebooks/qSNAP_example/example.ipynb)

# References

* **Gaussian Approximation Potential**: Bartók, A. P.; Payne, M. C.; Kondor, R.; Csányi, G. 
    Gaussian Approximation Potentials: The Accuracy of Quantum Mechanics, without the Electrons. 
    Physical Review Letters 2010, 104, 136403. 
    [doi:10.1103/PhysRevLett.104.136403](https://doi.org/10.1103/PhysRevLett.104.136403).
* **Moment Tensor Potential**: Shapeev, A. V. 
    Moment tensor potentials: A class of systematically improvable interatomic potentials. 
    Multiscale Modeling & Simulation, 14(3), 1153-1173.
    [doi:10.1137/15M1054183](https://epubs.siam.org/doi/abs/10.1137/15M1054183)
* **Neural Network Potential**: Behler, J., & Parrinello, M. 
    Generalized neural-network representation of high-dimensional potential-energy surfaces. 
    Physical Review Letters 2007, 98, 146401.
    [doi:10.1103/PhysRevLett.98.146401](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401)
* **Spectral Neighbor Analysis Potential**: Thompson, A. P., Swiler, L. P., Trott, 
    C. R., Foiles, S. M., & Tucker, G. J.
    Spectral neighbor analysis method for automated generation of quantum-accurate 
    interatomic potentials. Journal of Computational Physics, 285, 316-330.
    [doi:10.1016/j.jcp.12.018](https://doi.org/10.1016/j.jcp.2014.12.018)
