# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import os
import json
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from pymatgen import Structure
from monty.json import MSONable

from mlearn.model.gaussian_process import GaussianProcessRegressionModel

class GaussianProcessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()

    def setUp(self):
        self.x_train = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        self.y_train = (self.x_train * np.sin(self.x_train)).ravel()

        class DummyDescriber():
            def describe(self, obj):
                pass

            def describe_all(self, n):
                return pd.DataFrame(n)
        self.gpr = GaussianProcessRegressionModel(describer=DummyDescriber(), \
                                                  kernel_category='RBF')

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.this_dir)
        shutil.rmtree(cls.test_dir)

    def test_fit_predict(self):
        self.gpr.fit(inputs=self.x_train, outputs=self.y_train)
        x_test = np.atleast_2d(np.linspace(0, 9, 1000)).T
        y_test = x_test * np.sin(x_test)
        y_pred, sigma = self.gpr.predict(x_test, return_std=True)
        upper_bound = y_pred + 1.96 * sigma
        lower_bound = y_pred - 1.96 * sigma
        self.assertTrue(np.all([l < y and y < u for u, y, l in\
                                zip(upper_bound, y_test, lower_bound)]))

if __name__ == "__main__":
    unittest.main()