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

from mlearn.model.linear_model import LinearModel

class LinearModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x_train = np.random.rand(10, 2)
        cls.coef = np.random.rand(2)
        cls.intercept = np.random.rand()
        cls.y_train = cls.x_train.dot(cls.coef) + cls.intercept

    def setUp(self):
        class DummyDescriber(MSONable):
            def describe(self, obj):
                pass

            def describe_all(self, n):
                return pd.DataFrame(n)

        self.lm = LinearModel(DummyDescriber())

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_fit_predict(self):
        self.lm.fit(inputs=self.x_train, outputs=self.y_train)
        x_test = np.random.rand(10, 2)
        y_test = x_test.dot(self.coef) + self.intercept
        y_pred = self.lm.predict(x_test)
        np.testing.assert_array_almost_equal(y_test, y_pred)
        np.testing.assert_array_almost_equal(self.coef, self.lm.coef)
        self.assertAlmostEqual(self.intercept, self.lm.intercept)

    def test_evaluate_fit(self):
        self.lm.fit(inputs=self.x_train, outputs=self.y_train)
        y_pred = self.lm.evaluate_fit()
        np.testing.assert_array_almost_equal(y_pred, self.y_train)

    def test_serialize(self):
        json_str = json.dumps(self.lm.as_dict())
        recover = LinearModel.from_dict(json.loads(json_str))
        self.assertIsNotNone(recover)

    def model_save_load(self):
        self.lm.save(os.path.join(self.test_dir, 'test_lm.save'))
        ori = self.lm.model.coef_
        self.lm.load(os.path.join(self.test_dir, 'test_lm.save'))
        loaded = self.lm.model.coef_
        self.assertAlmostEqual(ori, loaded)

if __name__ == "__main__":
    unittest.main()