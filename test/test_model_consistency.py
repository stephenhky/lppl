
import unittest

import numpy as np

from lppl.fit import LPPLModel


class TestConsistency(unittest.TestCase):
    def case1(self):
        param = {
            'A': 569.988,
            'B': -266.943,
            'C': 14.242,
            'phi': 4.1,
            'omega': 7.877,
            'z': 0.445,
            'tc': 10
        }
        lppl_model = LPPLModel.load_model_from_parameters(param)

        ts = np.linspace(0, 12, 121)
        prices =lppl_model(ts)

        fitted_lppl_model = LPPLModel()
        fitted_lppl_model.fit(ts, prices)
        print('A: {}'.format(fitted_lppl_model.A))
        print('B: {}'.format(fitted_lppl_model.B))
        print('C: {}'.format(fitted_lppl_model.C))
        print('phi: {}'.format(fitted_lppl_model.phi))
        print('m: {}'.format(fitted_lppl_model.m))
        print('tc: {}'.format(fitted_lppl_model.tc))




if __name__ == '__main__':
    unittest.main()
