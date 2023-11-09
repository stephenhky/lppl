
from types import LambdaType
from typing import Tuple
from math import cos, log

import numpy as np
import numpy.typing as npt


def lppl_logprice_function(
        tc: float,
        m: float,
        omega: float,
        A: float,
        B: float,
        C: float,
        phi: float
) -> np.vectorize:
    # logpfcn = lambda t: A + (tc-t)**m * (B + C*cos(omega*log(tc-t)-phi))
    def logpfcn(t: float) -> float:
        # print("A={}; B={}; C={}; phi={}".format(A, B, C, phi))
        # print("tc={}; m={}; omega={}; phi={}; tc-t={}".format(tc, m, omega, phi, tc-t))
        return A + (tc-t)**m * (B + C*cos(omega*log(tc-t)-phi))
    return np.vectorize(logpfcn)


def lppl_costfunction(
        ts: npt.NDArray[np.float64],
        logprices: npt.NDArray[np.float64]
) -> LambdaType:
    def f(
            tc: float,
            m: float,
            omega: float,
            A: float,
            B: float,
            C: float,
            phi: float
    ) -> np.float64:
        log_price_func = lppl_logprice_function(tc, m, omega, A, B, C, phi)
        return np.sum(np.square(logprices - log_price_func(ts)))

    return f


def _lppl_syseqn_matrix(
        ts: npt.NDArray[np.float64],
        logprices: npt.NDArray[np.float64],
        tc: float,
        m: float,
        omega: float
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    syseqns_matrix = np.zeros((4, 4))
    syseqns_b = np.zeros(4)
    N = ts.shape[0]
    assert logprices.shape[0] == N

    deltat_pow_m = (tc-ts)**m

    syseqns_b[0] = np.sum(logprices)
    syseqns_b[1] = np.sum(deltat_pow_m * logprices)
    syseqns_b[2] = np.sum(deltat_pow_m * np.cos(omega * np.log(tc-ts)) * logprices)
    syseqns_b[3] = np.sum(deltat_pow_m * np.sin(omega * np.log(tc-ts)) * logprices)

    syseqns_matrix[0, 0] = N
    syseqns_matrix[0, 1] = np.sum(deltat_pow_m)
    syseqns_matrix[0, 2] = np.sum(deltat_pow_m * np.cos(omega * np.log(tc-ts)))
    syseqns_matrix[0, 3] = np.sum(deltat_pow_m * np.sin(omega * np.log(tc-ts)))

    syseqns_matrix[1, 0] = np.sum(deltat_pow_m)
    syseqns_matrix[1, 1] = np.sum(np.square(deltat_pow_m))
    syseqns_matrix[1, 2] = np.sum(np.square(deltat_pow_m) * np.cos(omega * np.log(tc-ts)))
    syseqns_matrix[1, 3] = np.sum(np.square(deltat_pow_m) * np.sin(omega * np.log(tc-ts)))

    syseqns_matrix[2, 0] = np.sum(deltat_pow_m * np.cos(omega * np.log(tc-ts)))
    syseqns_matrix[2, 1] = np.sum(np.square(deltat_pow_m) * np.cos(omega * np.log(tc-ts)))
    syseqns_matrix[2, 2] = np.sum(np.square(deltat_pow_m * np.cos(omega * np.log(tc-ts))))
    syseqns_matrix[2, 3] = np.sum(np.square(deltat_pow_m) * np.cos(omega * np.log(tc-ts)) * np.sin(omega * np.log(tc-ts)))

    syseqns_matrix[3, 0] = np.sum(deltat_pow_m * np.sin(omega * np.log(tc-ts)))
    syseqns_matrix[3, 1] = np.sum(np.square(deltat_pow_m) * np.sin(omega * np.log(tc-ts)))
    syseqns_matrix[3, 2] = np.sum(np.square(deltat_pow_m) * np.cos(omega * np.log(tc-ts)) * np.sin(omega * np.log(tc-ts)))
    syseqns_matrix[3, 3] = np.sum(np.square(deltat_pow_m * np.sin(omega * np.log(tc-ts))))

    return syseqns_matrix, syseqns_b


def _lppl_slaved_costfunction(
        ts: npt.NDArray[np.float64],
        logprices: npt.NDArray[np.float64],
) -> LambdaType:
    def f(
            tc: float,
            m: float,
            omega: float
    ) -> np.float64:
        # print('tc={}; m={}; omega={}'.format(tc, m, omega))

        cost_func = lppl_costfunction(ts, logprices)
        lineqn_matrix, b = _lppl_syseqn_matrix(ts, logprices, tc, m, omega)
        x = np.linalg.solve(lineqn_matrix, b)
        # print('sol: {}'.format(x))
        return cost_func(tc, m, omega, x[0], x[1], x[2], x[3])

    return f

