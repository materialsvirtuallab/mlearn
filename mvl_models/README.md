# Pre-trained ML-IAPs from Materials Virtual Lab

This directory contains pre-trained ML-IAPs from the Materials Virtual Lab. 

Currently we provide GAP, NNP, SNAP, and qSNAP ML-IAPs for Ni, Cu, Li, Mo, Si, and Ge systems.
For users who are interested in using these ML-IAPs for material properties prediction. 
(Make sure the LAMMPS-based interfaces are installed properly),
below are examples of material properties calculator usage with ML-IAPs.

Within `Ni/gap` directory,

```python
from mlearn.potentials.gap import GAPotential
from pymatgen import Structure, Element
from mlearn.potentials.lammps.calcs import EnergyForceStress, ElasticConstant

gap = GAPotential.from_config(filename='gap.xml')
gap.specie = Element('Ni')

Ni_conventional_cell = Structure.from_file('Ni_conventional.cif')
efs_calculator = EnergyForceStress(ff_settings=gap)
energy, forces, stresses = efs_calculator.calculate([Ni_conventional_cell])[0]
print('The predicted energy of Ni conventional cell is {} eV'.format(energy))
print('The predicted forces of Ni conventional cell is {} eV/Angstrom'.format(forces))

elastic_calculator = ElasticConstant(ff_settings=gap, lattice='fcc', alat=3.508)
C11, C12, C44, bulkmodulus = elastic_calculator.calculate()
print('The predicted C11, C12, C44, bulkmodulus are {}, {}, {}, {} GPa'.format(C11, C12, C44, bulkmodulus))
```

Within `Ni/nnp` directory,

```python
from mlearn.potentials.nnp import NNPotential
from pymatgen import Structure
from mlearn.potentials.lammps.calcs import EnergyForceStress, ElasticConstant

nnp = NNPotential.from_config(input_filename='input.nn', scaling_filename='scaling.data', weights_filename='weights.028.data')

Ni_conventional_cell = Structure.from_file('Ni_conventional.cif')
efs_calculator = EnergyForceStress(ff_settings=nnp)
energy, forces, stresses = efs_calculator.calculate([Ni_conventional_cell])[0]
print('The predicted energy of Ni conventional cell is {} eV'.format(energy))
print('The predicted forces of Ni conventional cell is {} eV/Angstrom'.format(forces))

elastic_calculator = ElasticConstant(ff_settings=nnp, lattice='fcc', alat=3.508)
C11, C12, C44, bulkmodulus = elastic_calculator.calculate()
print('The predicted C11, C12, C44, bulkmodulus are {}, {}, {}, {} GPa'.format(C11, C12, C44, bulkmodulus))
```

Within `Ni/snap` directory,

```python
from mlearn.potentials.snap import SNAPotential
from pymatgen import Structure, Element
from mlearn.potentials.lammps.calcs import EnergyForceStress, ElasticConstant

snap = SNAPotential.from_config(coeff_file='SNAPotential.snapcoeff', param_file='SNAPotential.snapparam')

Ni_conventional_cell = Structure.from_file('Ni_conventional.cif')
efs_calculator = EnergyForceStress(ff_settings=snap)
energy, forces, stresses = efs_calculator.calculate([Ni_conventional_cell])[0]
print('The predicted energy of Ni conventional cell is {} eV'.format(energy))
print('The predicted forces of Ni conventional cell is {} eV/Angstrom'.format(forces))

elastic_calculator = ElasticConstant(ff_settings=snap, lattice='fcc', alat=3.508)
C11, C12, C44, bulkmodulus = elastic_calculator.calculate()
print('The predicted C11, C12, C44, bulkmodulus are {}, {}, {}, {} GPa'.format(C11, C12, C44, bulkmodulus))
```

Within `Ni/qsnap` directory,

```python
from mlearn.potentials.snap import SNAPotential
from pymatgen import Structure, Element
from mlearn.potentials.lammps.calcs import EnergyForceStress, ElasticConstant

qsnap = SNAPotential.from_config(coeff_file='SNAPotential.snapcoeff', param_file='SNAPotential.snapparam')

Ni_conventional_cell = Structure.from_file('Ni_conventional.cif')
efs_calculator = EnergyForceStress(ff_settings=qsnap)
energy, forces, stresses = efs_calculator.calculate([Ni_conventional_cell])[0]
print('The predicted energy of Ni conventional cell is {} eV'.format(energy))
print('The predicted forces of Ni conventional cell is {} eV/Angstrom'.format(forces))

elastic_calculator = ElasticConstant(ff_settings=qsnap, lattice='fcc', alat=3.508)
C11, C12, C44, bulkmodulus = elastic_calculator.calculate()
print('The predicted C11, C12, C44, bulkmodulus are {}, {}, {}, {} GPa'.format(C11, C12, C44, bulkmodulus))
```