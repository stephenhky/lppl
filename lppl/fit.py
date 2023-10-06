
import numpy as np
from scipy.optimize import minimize

from .model import lppl_logprice_function, _lppl_slaved_costfunction


class LPPLModel:
    def __init__(self):
        self._fitted = False

    def fit(self, ts: np.typing.NDArray[np.float64], prices: np.typing.NDArray[np.float64]):
        assert ts.shape[0] == prices.shape[0]

        logprices = np.log(prices)

        slaved_costfunction = _lppl_slaved_costfunction(ts, logprices)
