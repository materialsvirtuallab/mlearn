# Installation of Descriptor Codes

Here are the installation instructions for the codes implementing the various
descriptors.

## Gaussian Approximation Potential (GAP)

Clone the [QUIP](https://github.com/libAtoms/QUIP) repository:

```
git clone --recursive https://github.com/libAtoms/QUIP.git
```
Define the environment variable `QUIP_ROOT` as the path of QUIP repository.
```
cd QUIP
export QUIP_ROOT=${PWD}
```
Decide your architecture by looking in the `${QUIP_ROOT}/arch/` directory and define the 
environmental variable `QUIP_ARCH`, e.g,
```
export QUIP_ARCH=linux_x86_64_gfortran
```
Make sure `QUIP_ROOT` and `QUIP_ARCH` are in environment path for further 
auto-detection of building LAMMPS.
 
Define the environment variable `QUIP_INSTALLDIR` and add it to the environment path. 
When the compilation and installation finishes, all compiled programs will be copied 
to `QUIP_INSTALLDIR`.
```
export QUIP_INSTALLDIR=<path_for_executables>
```
Obtain the support for Gaussian Approximation Potential from 
[GAP](http://www.libatoms.org/gap/gap_download.html), extract the codes in 
`${QUIP_ROOT}\src` directory.

Customise QUIP, set the maths libraries and provide linking options. (It will ask some 
questions, make sure answer yes to compile with **GAP** prediction and training support).
```
make config
```

Compile all programs, modules and libraries.
```
make
```

Copies all compiled programs to `QUIP_INSTALLDIR`.
```
make install
```

Compile QUIP as a library and link to it, this will create a library 
`build/${QUIP_ARCH}/libquip.a`.
```
make libquip
```

Clone the LAMMPS repository from GitHub. (if LAMMPS package has already been installed, 
skip this step).
```
git clone git@github.com:lammps/lammps.git
```

Assume `LMP_ROOT` is the path of LAMMPS package, enable the interface neccessary to 
use QUIP potentials in `${LMP_ROOT}/src` directory.
```
make yes-user-quip
```

The serial version of LAMMPS containing QUIP interface can then be installed. 
(Please refer to [LAMMPS website](https://lammps.sandia.gov/) for the installation 
of other versions.)
```
make serial
```

## Moment Tensor Potential (MTP)**

Clone the [MLIP](http://gitlab.skoltech.ru/shapeev/mlip-dev) repository. This 
repository is under development, please refer to 
[Alex Shapeev](http://www.shapeev.com) for access to the repository.
```
git clone http://gitlab.skoltech.ru/shapeev/mlip-dev.git
```

Assume `MLIP_ROOT` is the path of MLIP package. Configure the MLIP package 
build in `${MLIP_ROOT}` directory.
```
./configure
```

Compile and build the MLIP executable and add the `${MLIP_ROOT}/bin` to 
environment path.
```
make mlp
```

Clone the LAMMPS repository from GitHub. (if LAMMPS package has already been
installed, skip this step).
```
git clone git@github.com:lammps/lammps.git
```

Assume `LMP_ROOT` is the path of LAMMPS package. Configure MLIP to be used
within LAMMPS in `${MLIP_ROOT}` directory.
```
./configure --lammps=${LMP_ROOT}
```

Install LAMMPS with MLIP interface in `${MLIP_ROOT}` directory.
```
make lammps
```
An example is provided in `mlearn/notebooks/MTP_example/example.ipynb`.

## Neural Network Potential (NNP)

Clone the [n2p2](https://github.com/CompPhysVienna/n2p2) package from GitHub.
```
git clone https://github.com/CompPhysVienna/n2p2.git
```

Assume `N2P2_ROOT` is the path of the n2p2 package. Build the neccessary executables in
`${N2P2_ROOT}/src` directory. (Make sure 
[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) and 
[GSL](https://www.gnu.org/software/gsl) are installed and visible in the system).
```
make
```

Add the path of `${N2P2_ROOT}/bin` to the environment path.

Build the shared libraries of neural network potentials in `${N2P2_ROOT}/src` directory.
```
make libnnpif-shared
``` 

Add the path of NNP libraries `${N2P2_ROOT}/lib` to library path `LD_LIBRARY_PATH`.

Clone the LAMMPS repository from GitHub. (if LAMMPS package has already been installed, 
skip this step).
```
git clone git@github.com:lammps/lammps.git
```

Assume `LMP_ROOT` is the path of LAMMPS package. Link to NNP libraries in `${LMP_ROOT}` 
directory.
```
ln -s ${N2P2_ROOT} lib/nnp
```

Copy the USER-NNP package to the LAMMPS source directory.
```
cp -r ${N2P2_ROOT}/src/interface/LAMMPS/src/USER-NNP ${LMP_ROOT}/src
```

Enable LAMMPS interface with NNP in `${LMP_PATH}/src` directory.
```
make yes-user-nnp
```

The serial version of LAMMPS containing NNP interface can then be installed. 
(Please refer to [LAMMPS website](https://lammps.sandia.gov/) for the installation 
of other versions.)
```
make serial
```

## Spectral Neighbor Analysis Potential (SNAP)**

The calculations rely on [LAMMPS package](https://lammps.sandia.gov) itself. 
Assume `LMP_ROOT` is the path of LAMMPS package. To install SNAP package, enable 
the SNAP interface in `${LMP_ROOT}/src` directory.
```
make yes-snap
```

The serial version of LAMMPS containing SNAP interface can then be installed.
(Please refer to [LAMMPS website](https://lammps.sandia.gov/) for the installation 
of other versions.)
```
make serial
``` 