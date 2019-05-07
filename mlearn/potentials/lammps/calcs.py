# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import os
import abc
import io
import subprocess
import itertools

import six
import numpy as np
from monty.tempfile import ScratchDir
from mlearn.potentials import Potential
from pymatgen.io.lammps.data import LammpsData
from pymatgen import Structure, Lattice, Element


_sort_elements = lambda symbols: [e.symbol for e in
                                  sorted([Element(e) for e in symbols])]


def _pretty_input(lines):
    clean_lines = [l.strip('\n') for l in lines]
    commands = [l for l in clean_lines if len(l.strip()) > 0]
    keys = [c.split()[0] for c in commands
            if not c.split()[0].startswith('#')]
    width = max([len(k) for k in keys]) + 4
    prettify = lambda l: l.split()[0].ljust(width) + ' '.join(l.split()[1:]) \
        if not (len(l.split()) == 0 or l.strip().startswith('#')) else l
    new_lines = map(prettify, clean_lines)
    return '\n'.join(new_lines)


def _read_dump(file_name, dtype='float_'):
    with open(file_name) as f:
        lines = f.readlines()[9:]
    return np.loadtxt(io.StringIO(''.join(lines)), dtype=dtype)


class LMPStaticCalculator(six.with_metaclass(abc.ABCMeta, object)):
    """
    Abstract class to perform static structure property calculation
    using LAMMPS.

    """

    LMP_EXE = 'lmp_serial'
    _COMMON_CMDS = ['units metal',
                    'atom_style charge',
                    'box tilt large',
                    'read_data data.static',
                    'run 0']

    @abc.abstractmethod
    def _setup(self):
        """
        Setup a calculation, writing input files, etc.

        """
        return

    @abc.abstractmethod
    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return

    @abc.abstractmethod
    def _parse(self):
        """
        Parse results from dump files.

        """
        return

    def calculate(self, structures):
        """
        Perform the calculation on a series of structures.

        Args:
            structures [Structure]: Input structures in a list.

        Returns:
            List of computed data corresponding to each structure,
            varies with different subclasses.

        """
        for s in structures:
            assert self._sanity_check(s) is True, \
                'Incompatible structure found'
        ff_elements = None
        if hasattr(self, 'element_profile'):
            ff_elements = self.element_profile.keys()
        with ScratchDir('.'):
            input_file = self._setup()
            data = []
            for s in structures:
                ld = LammpsData.from_structure(s, ff_elements)
                ld.write_file('data.static')
                p = subprocess.Popen([self.LMP_EXE, '-in', input_file],
                                     stdout=subprocess.PIPE)
                stdout = p.communicate()[0]
                rc = p.returncode
                if rc != 0:
                    error_msg = 'LAMMPS exited with return code %d' % rc
                    msg = stdout.decode("utf-8").split('\n')[:-1]
                    try:
                        error_line = [i for i, m in enumerate(msg)
                                      if m.startswith('ERROR')][0]
                        error_msg += ', '.join([e for e in msg[error_line:]])
                    except:
                        error_msg += msg[-1]
                    raise RuntimeError(error_msg)
                results = self._parse()
                data.append(results)
        return data


class EnergyForceStress(LMPStaticCalculator):
    """
    Calculate energy, forces and virial stress of structures.
    """
    def __init__(self, ff_settings):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
        """
        self.ff_settings = ff_settings

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'efs')
        with open(os.path.join(template_dir, 'in.efs'), 'r') as f:
            input_template = f.read()

        input_file = 'in.efs'

        if isinstance(self.ff_settings, Potential):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings

        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(ff_settings)))
        return input_file

    def _sanity_check(self, structure):
        return True

    def _parse(self):
        energy = float(np.loadtxt('energy.txt'))
        force = _read_dump('force.dump')
        stress = np.loadtxt('stress.txt')
        return energy, force, stress


class SpectralNeighborAnalysis(LMPStaticCalculator):
    """
    Calculator for bispectrum components to characterize the local
    neighborhood of each atom in a general way.

    Usage:
        [(b, db, e)] = sna.calculate([Structure])
        b: 2d NumPy array with shape (N, n_bs) containing bispectrum
            coefficients, where N is the No. of atoms in structure and
            n_bs is the No. of bispectrum components.
        db: 2d NumPy array with shape (N, 3 * n_bs * n_elements)
            containing the first order derivatives of bispectrum
            coefficients with respect to atomic coordinates,
            where n_elements is the No. of elements in element_profile.
        e: 2d NumPy array with shape (N, 1) containing the element of
            each atom.

    """

    _CMDS = ['pair_style lj/cut 10',
             'pair_coeff * * 1 1',
             'compute sna all sna/atom ',
             'compute snad all snad/atom ',
             'compute snav all snav/atom ',
             'dump 1 all custom 1 dump.element element',
             'dump 2 all custom 1 dump.sna c_sna[*]',
             'dump 3 all custom 1 dump.snad c_snad[*]',
             'dump 4 all custom 1 dump.snav c_snav[*]']

    def __init__(self, rcutfac, twojmax, element_profile, rfac0=0.99363,
                 rmin0=0, diagonalstyle=3, quadratic=False):
        """
        For more details on the parameters, please refer to the
        official documentation of LAMMPS.

        Notes:
            Despite this calculator uses compute sna(d)/atom command
            (http://lammps.sandia.gov/doc/compute_sna_atom.html), the
            parameter definition is in consistent with pair_style snap
            document (http://lammps.sandia.gov/doc/pair_snap.html),
            where *rcutfac* is the cutoff in distance rather than some
            scale factor.

        Args:
            rcutfac (float): Global cutoff distance.
            twojmax (int): Band limit for bispectrum components.
            element_profile (dict): Parameters (cutoff factor 'r' and
                weight 'w') related to each element, e.g.,
                {'Na': {'r': 0.3, 'w': 0.9},
                 'Cl': {'r': 0.7, 'w': 3.0}}
            rfac0 (float): Parameter in distance to angle conversion.
                Set between (0, 1), default to 0.99363.
            rmin0 (float): Parameter in distance to angle conversion.
                Default to 0.
            diagonalstyle (int): Parameter defining which bispectrum
                components are generated. Choose among 0, 1, 2 and 3,
                default to 3.
            quadratic (bool): Whether including quadratic terms.
                Default to False.

        """
        self.rcutfac = rcutfac
        self.twojmax = twojmax
        self.element_profile = element_profile
        self.rfac0 = rfac0
        self.rmin0 = rmin0
        assert diagonalstyle in range(4), 'Invalid diagonalstype, ' \
                                          'choose among 0, 1, 2 and 3'
        self.diagonalstyle = diagonalstyle
        self.quadratic = quadratic

    @staticmethod
    def get_bs_subscripts(twojmax, diagonal):
        """
        Method to list the subscripts 2j1, 2j2, 2j of bispectrum
        components.

        Args:
            twojmax (int): Band limit for bispectrum components.
            diagonal (int): Parameter defining which bispectrum
            components are generated. Choose among 0, 1, 2 and 3.

        Returns:
            List of all subscripts [2j1, 2j2, 2j].

        """
        subs = itertools.product(range(twojmax + 1), repeat=3)
        filters = [lambda x: True if x[0] >= x[1] else False]
        if diagonal == 2:
            filters.append(lambda x: True if x[0] == x[1] == x[2] else False)
        else:
            if diagonal == 1:
                filters.append(lambda x: True if x[0] == x[1] else False)
            elif diagonal == 3:
                filters.append(lambda x: True if x[2] >= x[0] else False)
            elif diagonal == 0:
                pass
            j_filter = lambda x: True if \
                x[2] in range(x[0] - x[1], min(twojmax, x[0] + x[1]) + 1, 2)\
                else False
            filters.append(j_filter)
        for f in filters:
            subs = filter(f, subs)
        return list(subs)

    @property
    def n_bs(self):
        """
        Returns No. of bispectrum components to be calculated.

        """
        return len(self.get_bs_subscripts(self.twojmax, self.diagonalstyle))

    def _setup(self):
        compute_args = '{} {} {} '.format(1, self.rfac0, self.twojmax)
        el_in_seq = _sort_elements(self.element_profile.keys())
        cutoffs = [self.element_profile[e]['r'] * self.rcutfac
                   for e in el_in_seq]
        weights = [self.element_profile[e]['w'] for e in el_in_seq]
        compute_args += ' '.join([str(p) for p in cutoffs + weights])
        qflag = 1 if self.quadratic else 0
        compute_args += ' diagonal {} rmin0 {} quadraticflag {}'.\
            format(self.diagonalstyle, self.rmin0, qflag)
        add_args = lambda l: l + compute_args if l.startswith('compute') \
            else l
        CMDS = list(map(add_args, self._CMDS))
        CMDS[2] += ' bzeroflag 0'
        CMDS[3] += ' bzeroflag 0'
        CMDS[4] += ' bzeroflag 0'
        dump_modify = 'dump_modify 1 element '
        dump_modify += ' '.join(str(e) for e in el_in_seq)
        CMDS.append(dump_modify)
        ALL_CMDS = self._COMMON_CMDS[:]
        ALL_CMDS[-1:-1] = CMDS
        input_file = 'in.sna'
        with open('in.sna', 'w') as f:
            f.write(_pretty_input(ALL_CMDS).format(self.twojmax, self.rfac0))
        return input_file

    def _sanity_check(self, structure):
        struc_elements = set(structure.symbol_set)
        sna_elements = self.element_profile.keys()
        return struc_elements.issubset(sna_elements)

    def _parse(self):
        element = np.atleast_1d(_read_dump('dump.element', 'unicode'))
        b = np.atleast_2d(_read_dump('dump.sna'))
        db = np.atleast_2d(_read_dump('dump.snad'))
        vb = np.atleast_2d(_read_dump('dump.snav'))
        return b, db, vb, element


class ElasticConstant(LMPStaticCalculator):
    """
    Elastic constant calculator.
    """
    _RESTART_CONFIG = {'internal': {'write_command': 'write_restart',
                                    'read_command': 'read_restart',
                                    'restart_file': 'restart.equil'},
                       'external': {'write_command': 'write_data',
                                    'read_command': 'read_data',
                                    'restart_file': 'data.static'}}
    def __init__(self, ff_settings, potential_type='external',
                 deformation_size=1e-6, jiggle=1e-5, lattice='bcc', alat=5.0,
                 maxiter=400, maxeval=1000):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            potential_type (str): 'internal' indicates the internal potentials
                installed in lammps, 'external' indicates the external potentials
                outside of lammps.
            deformation_size (float): Finite deformation size. Usually range from
                1e-2 to 1e-8, to confirm the results not depend on it.
            jiggle (float): The amount of random jiggle for atoms to
                prevent atoms from staying on saddle points.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
            maxiter (float): The maximum number of iteration. Default to 400.
            maxeval (float): The maximum number of evaluation. Default to 1000.
        """
        self.ff_settings = ff_settings
        self.write_command = self._RESTART_CONFIG[potential_type]['write_command']
        self.read_command = self._RESTART_CONFIG[potential_type]['read_command']
        self.restart_file = self._RESTART_CONFIG[potential_type]['restart_file']
        self.deformation_size = deformation_size
        self.jiggle = jiggle
        self.lattice = lattice
        self.alat = alat
        self.maxiter = maxiter
        self.maxeval = maxeval

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'elastic')

        with open(os.path.join(template_dir, 'in.elastic'), 'r') as f:
            input_template = f.read()
        with open(os.path.join(template_dir, 'init.template'), 'r') as f:
            init_template = f.read()
        with open(os.path.join(template_dir, 'potential.template'), 'r') as f:
            potential_template = f.read()
        with open(os.path.join(template_dir, 'displace.template'), 'r') as f:
            displace_template = f.read()

        input_file = 'in.elastic'

        if isinstance(self.ff_settings, Potential):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings

        with open(input_file, 'w') as f:
            f.write(input_template.format(write_restart=self.write_command,
                                          restart_file=self.restart_file))
        with open('init.mod', 'w') as f:
            f.write(init_template.format(deformation_size=self.deformation_size,
                                         jiggle=self.jiggle, maxiter=self.maxiter,
                                         maxeval=self.maxeval, lattice=self.lattice,
                                         alat=self.alat))
        with open('potential.mod', 'w') as f:
            f.write(potential_template.format(ff_settings='\n'.join(ff_settings)))
        with open('displace.mod', 'w') as f:
            f.write(displace_template.format(read_restart=self.read_command,
                                             restart_file=self.restart_file))
        return input_file

    def calculate(self):
        with ScratchDir('.'):
            input_file = self._setup()
            p = subprocess.Popen([self.LMP_EXE, '-in', input_file],
                                 stdout=subprocess.PIPE)
            stdout = p.communicate()[0]
            rc = p.returncode
            if rc != 0:
                error_msg = 'LAMMPS exited with return code %d' % rc
                msg = stdout.decode("utf-8").split('\n')[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg)
                                  if m.startswith('ERROR')][0]
                    error_msg += ', '.join([e for e in msg[error_line:]])
                except:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)
            result = self._parse()
        return result

    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return True

    def _parse(self):
        """
        Parse results from dump files.

        """
        C11, C12, C44, bulkmodulus = np.loadtxt('elastic.txt')
        return C11, C12, C44, bulkmodulus


class LatticeConstant(LMPStaticCalculator):
    """
    Lattice Constant Relaxation Calculator.
    """
    def __init__(self, ff_settings):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
        """
        self.ff_settings = ff_settings

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'latt')

        with open(os.path.join(template_dir, 'in.latt'), 'r') as f:
            input_template = f.read()

        input_file = 'in.latt'

        if isinstance(self.ff_settings, Potential):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings

        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(ff_settings)))

        return input_file

    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return True

    def _parse(self):
        """
        Parse results from dump files.

        """
        a, b, c = np.loadtxt('lattice.txt')
        return a, b, c

class NudgedElasticBand(LMPStaticCalculator):
    """
    NudgedElasticBand migration energy calculator.
    """
    def __init__(self, ff_settings, specie, lattice, alat, num_replicas=8):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
            num_replicas (int): Number of replicas to use.
        """
        self.ff_settings = ff_settings
        self.specie = specie
        self.lattice = lattice
        self.alat = alat
        self.num_replicas = num_replicas

    def get_unit_cell(self, specie, lattice, alat):
        """
        Get the unit cell from specie, lattice type and lattice constant.

        Args
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
        """
        if lattice == 'fcc':
            unit_cell = Structure.from_spacegroup(sg='Fm-3m',
                                                  lattice=Lattice.cubic(alat),
                                                  species=[specie], coords=[[0, 0, 0]])
        elif lattice == 'bcc':
            unit_cell = Structure.from_spacegroup(sg='Im-3m',
                                                  lattice=Lattice.cubic(alat),
                                                  species=[specie], coords=[[0, 0, 0]])
        elif lattice == 'diamond':
            unit_cell = Structure.from_spacegroup(sg='Fd-3m',
                                                  lattice=Lattice.cubic(alat),
                                                  species=[specie], coords=[[0, 0, 0]])
        else:
            raise ValueError("Lattice type is invalid.")

        return unit_cell

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'neb')

        with open(os.path.join(template_dir, 'in.relax.template'), 'r') as f:
            relax_template = f.read()
        with open(os.path.join(template_dir, 'in.neb.template'), 'r') as f:
            neb_template = f.read()

        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice,
                                       alat=self.alat)
        lattice_calculator = LatticeConstant(ff_settings=self.ff_settings)
        a, _, _ = lattice_calculator.calculate([unit_cell])[0]
        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice,
                                       alat=a)

        super_cell = unit_cell * [4, 4, 4]
        ld = LammpsData.from_structure(super_cell, atom_style='atomic')
        ld.write_file('initial.vac')

        if self.lattice == 'fcc':
            del_id = 43
            vacneigh_ids = [170, 171]
        elif self.lattice == 'bcc':
            del_id = 43
            vacneigh_ids = [90, 91]
        elif self.lattice == 'diamond':
            del_id = 407
            vacneigh_ids = [342, 214, 87, 471, 167, 283, 43]
        else:
            raise ValueError("Lattice type is invalid.")

        basis = '\n'.join(['                basis {} {} {}  &'.format(*site.frac_coords)
                           for site in unit_cell])
        with open('in.relax', 'w') as f:
            f.write(relax_template.format(alat=a, basis=basis, specie=self.specie,
                                          ff_settings='\n'.join(self.ff_settings.write_param()),
                                          del_id=del_id, vacneigh_ids=' '.join([str(idx)
                                          for idx in vacneigh_ids])))

        p = subprocess.Popen([self.LMP_EXE, '-in', 'in.relax'], stdout=subprocess.PIPE)
        stdout = p.communicate()[0]

        rc = p.returncode
        if rc != 0:
            error_msg = 'LAMMPS exited with return code %d' % rc
            msg = stdout.decode("utf-8").split('\n')[:-1]
            try:
                error_line = [i for i, m in enumerate(msg)
                              if m.startswith('ERROR')][0]
                error_msg += ', '.join([e for e in msg[error_line:]])
            except:
                error_msg += msg[-1]
            raise RuntimeError(error_msg)

        ld_relaxed = LammpsData.from_file('data.relaxed', atom_style='atomic')

        if self.lattice == 'fcc':
            lines = ['2']
            lines.append('{}  {} {} {}'.format(str(170), *super_cell[169].coords))
            lines.append('{}  {} {} {}'.format(str(171), *super_cell[42].coords))
            with open('final.vac', 'w') as f:
                f.write('\n'.join(lines))

        elif self.lattice == 'bcc':
            lines = ['2']
            lines.append('{}  {} {} {}'.format(str(90), *super_cell[89].coords))
            lines.append('{}  {} {} {}'.format(str(91), *super_cell[42].coords))
            with open('final.vac', 'w') as f:
                f.write('\n'.join(lines))

        elif self.lattice == 'diamond':
            lines = ['7']
            lines.append('{}  {} {} {}'.format(str(342), *super_cell[341].coords))
            lines.append('{}  {} {} {}'.format(str(214), *super_cell[213].coords))
            lines.append('{}  {} {} {}'.format(str(87), *super_cell[86].coords))
            frac_coords = np.concatenate(
                (ld_relaxed.structure[86].frac_coords[:2] + [0.0625, 0.0625],
                 super_cell[406].frac_coords[2] + [0.015]))
            lines.append('{}  {} {} {}'.format(str(471),
                        *ld_relaxed.structure.lattice.get_cartesian_coords(frac_coords)))
            frac_coords = np.concatenate(
                (ld_relaxed.structure[213].frac_coords[:2] + [0.0625, 0.0625],
                 ld_relaxed.structure[166].frac_coords[2] - [0.01]))
            lines.append('{}  {} {} {}'.format(str(167),
                        *ld_relaxed.structure.lattice.get_cartesian_coords(frac_coords)))
            frac_coords = np.concatenate(
                (ld_relaxed.structure[341].frac_coords[:2] + [0.0625, 0.0625],
                 ld_relaxed.structure[282].frac_coords[2] - [0.01]))
            lines.append('{}  {} {} {}'.format(str(283),
                        *ld_relaxed.structure.lattice.get_cartesian_coords(frac_coords)))
            frac_coords = np.concatenate(
                (ld_relaxed.structure[470].frac_coords[:2] + [0.0625, 0.0625],
                 ld_relaxed.structure[42].frac_coords[2] + [0.015]))
            lines.append('{}  {} {} {}'.format(str(43),
                        *ld_relaxed.structure.lattice.get_cartesian_coords(frac_coords)))

        else:
            raise ValueError("Lattice type is invalid.")

        with open('final.vac', 'w') as f:
            f.write('\n'.join(lines))

        input_file = 'in.neb'

        with open(input_file, 'w') as f:
            f.write(neb_template.format(alat=a, basis=basis, specie=self.specie,
                    ff_settings='\n'.join(self.ff_settings.write_param()),
                    del_id=del_id, vacneigh_ids=' '.join([str(idx) for idx in vacneigh_ids])))

        return input_file

    def calculate(self):
        with ScratchDir('.'):
            input_file = self._setup()
            p = subprocess.Popen(['mpirun', '-n', str(self.num_replicas),
                                  'lmp_mpi', '-partition', '{}x1'.format(self.num_replicas),
                                  '-in', input_file],
                                  stdout=subprocess.PIPE)
            stdout = p.communicate()[0]
            rc = p.returncode
            if rc != 0:
                error_msg = 'LAMMPS exited with return code %d' % rc
                msg = stdout.decode("utf-8").split('\n')[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg)
                                  if m.startswith('ERROR')][0]
                    error_msg += ', '.join([e for e in msg[error_line:]])
                except:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)
            result = self._parse()
        return result

    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return True

    def _parse(self):
        """
        Parse results from dump files.

        """
        migration_barrier = _read_dump('log.lammps')[-1][6]
        return migration_barrier