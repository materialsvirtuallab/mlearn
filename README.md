# mlearn

The mlearn package is a benchmark suite for machine learning interatomic 
potentials for materials science. It enables a seamless way to develop 
various potentials (
[Guassian Approximation Potential](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.104.136403), 
[Moment Tensor Potential](https://epubs.siam.org/doi/abs/10.1137/15M1054183), 
[Neural Network Potential](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401),
[Spectral Neighbor Analysis Potential](https://www.sciencedirect.com/science/article/pii/S0021999114008353)) 
and provides LAMMPS-driven properties predictor with developed potentials as plugins.

# Installation

The usage of mlearn requires installation of specific packages and the plugins 
in [LAMMPS](https://lammps.sandia.gov/). Please see 
[detaled installation instructions](docs/install.md) for all descriptors.

# Jupyter Notebook Examples

* [Gaussian Approximation Potential (GAP)](notebooks/GAP_example/example.ipynb)
* [Moment Tensor Potential (MTP)](notebooks/MTP_example/example.ipynb)
* [Neural Network Potential (NNP)](notebooks/NNP_example/example.ipynb)
* [Linear Spectral Neighbor Analysis Potential (SNAP)](notebooks/SNAP_example/example.ipynb) 
  and [quadratic SNAP](notebooks/qSNAP_example/example.ipynb)
  