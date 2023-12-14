
from typing import Union, IO
import json

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, Bounds

from .numerics import lppl_logprice_function, _lppl_slaved_costfunction, _lppl_syseqn_matrix


class LPPLModel:
    def __init__(self):
        self._fitted = False
        self._m_lo = 0.1
        self._m_hi = 0.9
        self._omega_lo = 6. / (24. * 3600)
        self._omega_hi = 13. / (24. * 3600)
        self._tcgap = 0.5

    def fit(self, ts: npt.NDArray[np.float64], prices: npt.NDArray[np.float64]):
        assert ts.shape[0] == prices.shape[0]

        logprices = np.log(prices)

        slaved_costfunction = _lppl_slaved_costfunction(ts, logprices)
        wr_slaved_costfunction = lambda x: slaved_costfunction(x[0], x[1], x[2])

        init_tc = np.max(ts) + self._tcgap * 100
        init_m = 0.5
        init_omega = 0.5 * (self._omega_lo + self._omega_hi)

        # solve for non-linear parameters
        bounds = Bounds(
            [np.max(ts) + self._tcgap, self._m_lo, self._omega_lo],
            [np.inf, self._m_hi, self._omega_hi]
        )
        sol = minimize(
            wr_slaved_costfunction,
            x0=np.array([init_tc, init_m, init_omega]),
            bounds=bounds,
            method='Nelder-Mead'
        )
        tc = sol.x[0]
        m = sol.x[1]
        omega = sol.x[2]

        # solve for linear parameters
        syseqn_matrix, b = _lppl_syseqn_matrix(ts, logprices, tc, m, omega)
        linX = np.linalg.solve(syseqn_matrix, b)
        A, B, C1, C2 = linX[0], linX[1], linX[2], linX[3]
        C = np.sqrt(C1*C1+C2*C2)
        phi = np.arctan(C2/C1)

        # making model
        self._tc = tc
        self._m = m
        self._omega = omega
        self._A = A
        self._B = B
        self._C = C
        self._phi = phi
        self._lppl_logprice_fcn = lppl_logprice_function(self._tc, self._m, self._omega, self._A, self._B, self._C, self._phi)

        self._fitted = True

    def __call__(self, t: Union[float, npt.NDArray[np.float64]]) -> Union[float, npt.NDArray[np.float64]]:
        if self._fitted:
            return np.exp(self._lppl_logprice_fcn(t))
        else:
            raise NotImplementedError()

    @property
    def tc(self) -> float:
        return self._tc

    @property
    def m(self) -> float:
        return self._m

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def A(self) -> float:
        return self._A

    @property
    def B(self) -> float:
        return self._B

    @property
    def C(self) -> float:
        return self._C

    @property
    def phi(self) -> float:
        return self._phi

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def m_lo(self) -> float:
        return self._m_lo

    @property
    def m_hi(self) -> float:
        return self._m_hi

    @property
    def omega_lo(self) -> float:
        return self._omega_lo

    @property
    def omega_hi(self) -> float:
        return self._omega_hi

    @property
    def tcgap(self) -> float:
        return self._tcgap

    @m_lo.setter
    def m_lo(self, value: float):
        self._m_lo = value

    @m_hi.setter
    def m_hi(self, value: float):
        self._m_hi = value

    @omega_lo.setter
    def omega_lo(self, value: float):
        self._omega_lo = value

    @omega_hi.setter
    def omega_hi(self, value: float):
        self._omega_hi = value

    @tcgap.setter
    def tcgap(self, value: float):
        self._tcgap = value

    def dump_model_parameters(self):
        return {
            'tc': self._tc,
            'm': self._m,
            'omega': self._omega,
            'A': self._A,
            'B': self._B,
            'C': self._C,
            'phi': self._phi
        }

    def dump_model_jsonfile(self, f: IO):
        json.dump(self.dump_model_parameters(), f)

    def summary(self) -> str:
        if self._fitted:
            summarytxt = "tc: {}".format(self._tc) + '\n'
            summarytxt += "m: {}".format(self.m) + '\n'
            summarytxt += "omega: {}".format(self._omega) + '\n'
            summarytxt += "A: {}".format(self._A) + '\n'
            summarytxt += "B: {}".format(self._B) + '\n'
            summarytxt += "C: {}".format(self._C) + '\n'
            summarytxt += "phi: {}".format(self._phi)
            return summarytxt
        else:
            return "Model not fitted."

    def __str__(self):
        return self.summary()

    @classmethod
    def load_model_from_parameters(cls, param: dict):
        model = cls()
        model._tc = param['tc']
        model._m = param['m']
        model._omega = param['omega']
        model._A = param['A']
        model._B = param['B']
        model._C = param['C']
        model._phi = param['phi']
        model._lppl_logprice_fcn = lppl_logprice_function(
            model._tc, model._m, model._omega, model._A, model._B, model._C, model._phi
        )
        model._fitted = True

        return model

    @classmethod
    def load_model_from_jsonfile(cls, f: IO):
        param = json.load(f)
        return cls.load_model_from_parameters(param)

# optimization ref: https://towardsdatascience.com/introduction-to-optimization-constraints-with-scipy-7abd44f6de25#5ca2

