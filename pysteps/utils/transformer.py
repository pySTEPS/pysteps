import numpy as np
import scipy.stats as scipy_stats
from scipy.interpolate import interp1d
from typing import Optional

class BaseTransformer:
    def __init__(self, threshold: float = 0.5, zerovalue: Optional[float] = None):
        self.threshold = threshold
        self.zerovalue = zerovalue
        self.metadata = {}

    def transform(self, R: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, R: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_metadata(self) -> dict:
        return self.metadata.copy()

class DBTransformer(BaseTransformer):
    """
    DBTransformer applies a thresholded dB transform to rain rate fields.

    Parameters:
        threshold (float): Rain rate threshold (in mm/h). Values below this are set to `zerovalue` in dB.
        zerovalue (Optional[float]): Value in dB space to assign below-threshold pixels. If None, defaults to log10(threshold) - 0.1
    """

    def __init__(self, threshold: float = 0.5, zerovalue: Optional[float] = None):
        super().__init__(threshold, zerovalue)
        threshold_db = 10.0 * np.log10(self.threshold)

        if self.zerovalue is None:
            self.zerovalue = threshold_db - 0.1

        self.metadata = {
            "transform": "dB",
            "threshold": self.threshold,      # stored in mm/h
            "zerovalue": self.zerovalue       # stored in dB
        }

    def transform(self, R: np.ndarray) -> np.ndarray:
        R = R.copy()
        mask = R < self.threshold
        R[~mask] = 10.0 * np.log10(R[~mask])
        R[mask] = self.zerovalue
        return R

    def inverse_transform(self, R: np.ndarray) -> np.ndarray:
        R = R.copy()
        R = 10.0 ** (R / 10.0)
        R[R < self.threshold] = 0
        return R


class BoxCoxTransformer(BaseTransformer):
    def __init__(self, Lambda: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.Lambda = Lambda

    def transform(self, R: np.ndarray) -> np.ndarray:
        R = R.copy()
        mask = R < self.threshold

        if self.Lambda == 0.0:
            R[~mask] = np.log(R[~mask])
            tval = np.log(self.threshold)
        else:
            R[~mask] = (R[~mask] ** self.Lambda - 1) / self.Lambda
            tval = (self.threshold ** self.Lambda - 1) / self.Lambda

        if self.zerovalue is None:
            self.zerovalue = tval - 1

        R[mask] = self.zerovalue

        self.metadata = {
            "transform": "BoxCox",
            "lambda": self.Lambda,
            "threshold": tval,
            "zerovalue": self.zerovalue,
        }
        return R

    def inverse_transform(self, R: np.ndarray) -> np.ndarray:
        R = R.copy()
        if self.Lambda == 0.0:
            R = np.exp(R)
        else:
            R = np.exp(np.log(self.Lambda * R + 1) / self.Lambda)

        threshold_inv = (
            np.exp(np.log(self.Lambda * self.metadata["threshold"] + 1) / self.Lambda)
            if self.Lambda != 0.0 else
            np.exp(self.metadata["threshold"])
        )

        R[R < threshold_inv] = self.metadata["zerovalue"]
        self.metadata["transform"] = None
        return R

class NQTransformer(BaseTransformer):
    def __init__(self, a: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self._inverse_interp = None

    def transform(self, R: np.ndarray) -> np.ndarray:
        R = R.copy()
        shape = R.shape
        R = R.ravel()
        mask = ~np.isnan(R)
        R_ = R[mask]

        n = R_.size
        Rpp = ((np.arange(n) + 1 - self.a) / (n + 1 - 2 * self.a))
        Rqn = scipy_stats.norm.ppf(Rpp)
        R_sorted = R_[np.argsort(R_)]
        R_trans = np.interp(R_, R_sorted, Rqn)

        self.zerovalue = np.min(R_)
        R_trans[R_ == self.zerovalue] = 0

        self._inverse_interp = interp1d(
            Rqn, R_sorted, bounds_error=False,
            fill_value=(float(R_sorted.min()), float(R_sorted.max())) # type: ignore
        )

        R[mask] = R_trans
        R = R.reshape(shape)

        self.metadata = {
            "transform": "NQT",
            "threshold": R_trans[R_trans > 0].min(),
            "zerovalue": 0,
        }
        return R

    def inverse_transform(self, R: np.ndarray) -> np.ndarray:
        if self._inverse_interp is None:
            raise RuntimeError("Must call transform() before inverse_transform()")

        R = R.copy()
        shape = R.shape
        R = R.ravel()
        mask = ~np.isnan(R)
        R[mask] = self._inverse_interp(R[mask])
        R = R.reshape(shape)

        self.metadata["transform"] = None
        return R
    
class SqrtTransformer(BaseTransformer):
    def transform(self, R: np.ndarray) -> np.ndarray:
        R = np.sqrt(R)
        self.metadata = {
            "transform": "sqrt",
            "threshold": np.sqrt(self.threshold),
            "zerovalue": np.sqrt(self.zerovalue) if self.zerovalue else 0.0
        }
        return R

    def inverse_transform(self, R: np.ndarray) -> np.ndarray:
        R = R**2
        self.metadata["transform"] = None
        return R

def get_transformer(name: str, **kwargs) -> BaseTransformer:
    name = name.lower()
    if name == "boxcox":
        return BoxCoxTransformer(**kwargs)
    elif name == "db":
        return DBTransformer(**kwargs)
    elif name == "nqt":
        return NQTransformer(**kwargs)
    elif name == "sqrt":
        return SqrtTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown transformer type: {name}")
