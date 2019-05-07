# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import os
import random
import unittest
import tempfile
import numpy as np
from monty.os.path import which
from pymatgen import Lattice, Structure, Element

from mlearn.describers import BispectrumCoefficients

class BispectrumCoefficientsTest(unittest.TestCase):

    @staticmethod
    def test_subscripts():

        def from_lmp_doc(twojmax, diagonal):
            js = []
            for j1 in range(0, twojmax + 1):
                if diagonal == 2:
                    js.append([j1, j1, j1])
                elif diagonal == 1:
                    for j in range(0, min(twojmax, 2 * j1) + 1, 2):
                        js.append([j1, j1, j])
                elif diagonal == 0:
                    for j2 in range(0, j1 + 1):
                        for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                            js.append([j1, j2, j])
                elif diagonal == 3:
                    for j2 in range(0, j1 + 1):
                        for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                            if j >= j1:
                                js.append([j1, j2, j])
            return js

        profile = {"Mo": {"r": 0.5, "w": 1}}
        for d in range(4):
            for tjm in range(11):
                bc = BispectrumCoefficients(1.0, twojmax=tjm,
                                            element_profile=profile,
                                            quadratic=False,
                                            diagonalstyle=d)
                np.testing.assert_equal(bc.subscripts, from_lmp_doc(tjm, d))

    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found")
    def test_describe(self):
        s = Structure.from_spacegroup(225, Lattice.cubic(5.69169),
                                      ['Na', 'Cl'],
                                      [[0, 0, 0], [0, 0, 0.5]])
        profile = dict(Na=dict(r=0.3, w=0.9),
                       Cl=dict(r=0.7, w=3.0))
        s *= [2, 2, 2]
        structures = [s] * 10
        for s in structures:
            n = np.random.randint(4)
            inds = np.random.randint(16, size=n)
            s.remove_sites(inds)

        bc_atom = BispectrumCoefficients(5, 3, profile, diagonalstyle=2,
                                         quadratic=False, pot_fit=False)
        df_atom = bc_atom.describe_all(structures)
        for i, s in enumerate(structures):
            df_s = df_atom.xs(i, level='input_index')
            self.assertEqual(df_s.shape, (len(s), 4))
            self.assertTrue(df_s.equals(bc_atom.describe(s)))

        bc_pot = BispectrumCoefficients(5, 3, profile, diagonalstyle=2,
                                        quadratic=False, pot_fit=True)
        df_pot = bc_pot.describe_all(structures, include_stress=True)
        for i, s in enumerate(structures):
            df_s = df_pot.xs(i, level='input_index')
            self.assertEqual(df_s.shape, ((1 + len(s) * 3 + 6, 10)))
            self.assertTrue(df_s.equals(bc_pot.describe(s, include_stress=True)))
            sna = df_s.iloc[0]
            for specie in ['Na', 'Cl']:
                self.assertAlmostEqual(
                    sna[specie, 'n'],
                    s.composition.fractional_composition[specie])
                np.testing.assert_array_equal(df_s[specie, 'n'][1:],
                                              np.zeros(len(s) * 3 + 6))

if __name__ == "__main__":
    unittest.main()