
from typing import Union, IO
import json

import numpy as np
from scipy.optimize import minimize

from .model import lppl_logprice_function, _lppl_slaved_costfunction, _lppl_syseqn_matrix


class LPPLModel:
    def __init__(self):
        self._fitted = False

    def fit(self, ts: np.typing.NDArray[np.float64], prices: np.typing.NDArray[np.float64]):
        assert ts.shape[0] == prices.shape[0]

        logprices = np.log(prices)

        slaved_costfunction = _lppl_slaved_costfunction(ts, logprices)
        wr_slaved_costfunction = lambda x: slaved_costfunction(x[0], x[1], x[2])

        init_tc = np.max(ts)
        init_m = 1.
        init_omega = 1.

        # solve for non-linear parameters
        sol = minimize(wr_slaved_costfunction, x0=np.array([init_tc, init_m, init_omega]), method='Nelder-Mead')
        tc = sol.x[0]
        m = sol.x[1]
        omega = sol.x[2]

        # solve for linear parameters
        syseqn_matrix, b = _lppl_syseqn_matrix(ts, logprices, tc, m, omega)
        linX = np.linalg.solve(syseqn_matrix)
        A, B, C1, C2 = linX[0], linX[1], linX[2], linX[3]
        C = np.sqrt(C1*C1+C2*C2)
        phi = np.atan(C2/C1)

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

    def __call__(self, t: Union[float, np.typing.NDArray[np.float64]]) -> Union[float, np.typing.NDArray[np.float64]]:
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
