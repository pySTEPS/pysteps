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
            "threshold": self.threshold,  # stored in mm/h
            "zerovalue": self.zerovalue,  # stored in dB
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
            tval = (self.threshold**self.Lambda - 1) / self.Lambda

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
        tval = self.metadata["threshold"]
        zeroval = self.metadata["zerovalue"]

        # mask of values that were transformed (>= threshold in transformed space)
        m = R >= tval

        if self.Lambda == 0.0:
            R[m] = np.exp(R[m])
        else:
            # safe: we're not touching the below-threshold cells
            R[m] = np.exp(np.log(self.Lambda * R[m] + 1.0) / self.Lambda)

        # below-threshold cells get filled to zerovalue directly
        R[~m] = zeroval

        self.metadata["transform"] = None
        return R


class NQTransformer(BaseTransformer):
    def __init__(self, a: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self._inverse_interp = None
        self._qmin = None
        self._qmax = None
        self._min_value = None
        self._zero_token = 0.0

    def transform(self, R: np.ndarray) -> np.ndarray:
        R = R.copy()
        shape = R.shape
        R = R.ravel()
        mask = ~np.isnan(R)
        R_ = R[mask]

        n = R_.size
        Rpp = (np.arange(n) + 1 - self.a) / (n + 1 - 2 * self.a)
        Rqn = scipy_stats.norm.ppf(Rpp)
        order = np.argsort(R_)
        R_sorted = R_[order]
        R_trans = np.interp(R_, R_sorted, Rqn)

        # Record and map the minimum to zero (token)
        self._min_value = float(R_sorted[0])
        self.zerovalue = self._min_value
        self._zero_token = 0.0
        R_trans[R_ == self._min_value] = self._zero_token

        # Build inverse; we'll clip inputs
        self._inverse_interp = interp1d(
            Rqn, R_sorted, bounds_error=False, fill_value="extrapolate"  # type: ignore
        )
        self._qmin = float(Rqn.min())
        self._qmax = float(Rqn.max())

        R[mask] = R_trans
        R = R.reshape(shape)

        # Metadata: threshold is the smallest positive transformed value
        pos = R_trans[R_trans > self._zero_token]
        self.metadata = {
            "transform": "NQT",
            "threshold": float(pos.min()) if pos.size else np.inf,
            "zerovalue": self._zero_token,
        }
        return R

    def inverse_transform(self, R: np.ndarray) -> np.ndarray:
        if self._inverse_interp is None:
            raise RuntimeError("Must call transform() before inverse_transform()")

        R = R.copy()
        shape = R.shape
        R = R.ravel()
        mask = ~np.isnan(R)

        vals = R[mask]

        # 1) Exact zeros (the token) must map back to the minimum original value
        zero_mask = np.isclose(vals, self._zero_token, atol=1e-12)

        # 2) For the rest, clip to valid quantile range and interpolate back
        to_inv = np.clip(vals[~zero_mask], self._qmin, self._qmax)
        inv_vals = np.empty_like(vals)
        inv_vals[zero_mask] = self._min_value
        inv_vals[~zero_mask] = self._inverse_interp(to_inv)

        R[mask] = inv_vals
        R = R.reshape(shape)

        self.metadata["transform"] = None
        return R


class SqrtTransformer(BaseTransformer):
    def transform(self, R: np.ndarray) -> np.ndarray:
        R = np.sqrt(R)
        self.metadata = {
            "transform": "sqrt",
            "threshold": np.sqrt(self.threshold),
            "zerovalue": np.sqrt(self.zerovalue) if self.zerovalue else 0.0,
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
