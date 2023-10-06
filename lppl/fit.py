
import numpy as np
from scipy.optimize import minimize
from typing import Union

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


