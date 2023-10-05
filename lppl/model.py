
from types import LambdaType
from math import cos, sin, log

import numpy as np


def lppl_logprice_function(
        tc: float,
        m: float,
        omega: float,
        A: float,
        B: float,
        C: float,
        phi: float
) -> np.vectorize:
    logpfcn = lambda t: A + (tc-t)**m * (B + C*cos(omega*log(tc-t)-phi))
    return np.vectorize(logpfcn)


def lppl_costfunction(
        ts: np.typing.NDArray[np.float64],
        logprices: np.typing.NDArray[np.float64],

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
        ts: np.typing.NDArray[np.float64],
        logprices: np.typing.NDArray[np.float64],
        tc: float,
        m: float,
        omega: float
):
    syseqns_matrix = np.zeros((4, 4))
    syseqns_b = np.zeros(4)
    N = ts.shape[0]
    assert logprices.shape[0] == N

    deltat_pow_m = (ts-tc)**m

    syseqns_b[0] = np.sum(logprices)
    syseqns_b[1] = np.sum(deltat_pow_m * logprices)
    syseqns_b[2] = np.sum(deltat_pow_m * cos(omega * log(tc-ts)) * logprices)
    syseqns_b[3] = np.sum(deltat_pow_m * sin(omega * log(tc-ts)) * logprices)

