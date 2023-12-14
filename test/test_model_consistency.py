
import unittest

import numpy as np

from lppl.fit import LPPLModel


class TestConsistency(unittest.TestCase):
    def test_case1(self):
        param = {
            'A': 1.0,
            'B': -0.4,
            'C': 0.5,
            'phi': 4.1,
            'omega': 7.877,
            'm': 0.445,
            'tc': 10.0
        }
        lppl_model = LPPLModel.load_model_from_parameters(param)

        ts = np.linspace(0, 9, 91)
        prices = lppl_model(ts)

        fitted_lppl_model = LPPLModel()
        fitted_lppl_model.fit(ts, prices)
        print('A: {}'.format(fitted_lppl_model.A))
        print('B: {}'.format(fitted_lppl_model.B))
        print('C: {}'.format(fitted_lppl_model.C))
        print('phi: {}'.format(fitted_lppl_model.phi))
        print('omega: {}'.format(fitted_lppl_model.omega))
        print('m: {}'.format(fitted_lppl_model.m))
        print('tc: {}'.format(fitted_lppl_model.tc))
        assert fitted_lppl_model.tc > 9




if __name__ == '__main__':
    unittest.main()
