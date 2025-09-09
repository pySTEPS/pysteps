# test_transformers.py
import numpy as np
import pytest
from pysteps.utils.transformer import (
    BaseTransformer,
    DBTransformer,
    BoxCoxTransformer,
    NQTransformer,
    SqrtTransformer,
    get_transformer,
)

# ---------------------------
# Base class behavior
# ---------------------------


def test_base_transformer_not_implemented():
    bt = BaseTransformer()
    with pytest.raises(NotImplementedError):
        bt.transform(np.array([1.0]))
    with pytest.raises(NotImplementedError):
        bt.inverse_transform(np.array([1.0]))


def test_get_metadata_returns_copy():
    bt = BaseTransformer()
    bt.metadata = {"a": 1}
    meta = bt.get_metadata()
    meta["a"] = 2
    # internal metadata should remain unchanged
    assert bt.metadata["a"] == 1


# ---------------------------
# DBTransformer
# ---------------------------


def test_db_transformer_defaults_and_metadata():
    t = DBTransformer(threshold=0.5)
    expected_zero = 10.0 * np.log10(0.5) - 0.1
    assert np.isclose(t.zerovalue, expected_zero)
    meta = t.get_metadata()
    assert meta["transform"] == "dB"
    assert meta["threshold"] == 0.5
    assert np.isclose(meta["zerovalue"], expected_zero)


def test_db_transform_and_inverse_roundtrip():
    t = DBTransformer(threshold=0.5)
    R = np.array([0.0, 0.2, 0.5, 1.0, 2.0], dtype=float)
    Rt = t.transform(R)

    # Below threshold should be set to zerovalue (in dB)
    assert np.allclose(Rt[:2], t.zerovalue)

    # At/above threshold should be 10*log10
    assert np.isclose(Rt[2], 10 * np.log10(0.5))
    assert np.isclose(Rt[3], 10 * np.log10(1.0))
    assert np.isclose(Rt[4], 10 * np.log10(2.0))

    # Inverse: below-threshold become 0; others recover original
    Rin = t.inverse_transform(Rt)
    assert Rin[0] == 0.0
    assert Rin[1] == 0.0
    assert np.isclose(Rin[2], 0.5)
    assert np.isclose(Rin[3], 1.0)
    assert np.isclose(Rin[4], 2.0)


# ---------------------------
# BoxCoxTransformer
# ---------------------------


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.5])
def test_boxcox_transform_metadata_and_roundtrip(lam):
    t = BoxCoxTransformer(Lambda=lam, threshold=0.5)
    R = np.array([0.1, 0.3, 0.5, 1.0, 2.0], dtype=float)
    Rt = t.transform(R)

    # Metadata populated
    meta = t.get_metadata()
    assert meta["transform"] == "BoxCox"
    assert meta["lambda"] == lam
    assert "threshold" in meta and "zerovalue" in meta

    # Below threshold mapped to zerovalue
    assert np.allclose(Rt[R < 0.5], meta["zerovalue"])

    # Inverse should produce values >= threshold_inv for those above threshold
    Rin = t.inverse_transform(Rt)

    # After inverse, metadata["transform"] should be None
    assert t.metadata["transform"] is None

    # Values at/above threshold should be close to original (within numerical tolerance)
    mask = R >= 0.5
    assert np.allclose(Rin[mask], R[mask], rtol=1e-6, atol=1e-8)

    # Values below threshold get set to zerovalue by inverse (as per implementation)
    # i.e., not a true round-trip for below-threshold data (information is lost intentionally)
    assert np.allclose(Rin[~mask], t.metadata["zerovalue"])


def test_boxcox_with_explicit_zerovalue():
    t = BoxCoxTransformer(Lambda=0.0, threshold=0.5, zerovalue=-999.0)
    R = np.array([0.2, 0.6])
    Rt = t.transform(R)
    assert np.allclose(Rt[0], -999.0)
    assert t.metadata["zerovalue"] == -999.0


# ---------------------------
# NQTransformer
# ---------------------------


def test_nqt_requires_transform_before_inverse():
    t = NQTransformer()
    with pytest.raises(RuntimeError):
        t.inverse_transform(np.array([0.0]))


def test_nqt_transform_inverse_roundtrip_and_nans_preserved():
    rng = np.random.default_rng(42)
    R = rng.uniform(0.1, 10.0, size=(20,)).astype(float)
    R[[3, 7]] = np.nan  # include NaNs
    t = NQTransformer(a=0.0)
    Rt = t.transform(R)

    # Metadata populated
    meta = t.get_metadata()
    assert meta["transform"] == "NQT"
    assert meta["zerovalue"] == 0
    assert meta["threshold"] > 0  # min positive transformed value

    # NaNs preserved after transform
    assert np.isnan(Rt[3]) and np.isnan(Rt[7])

    # Inverse round-trip: finite values should come back ~original
    Rin = t.inverse_transform(Rt)
    assert t.metadata["transform"] is None

    mask = ~np.isnan(R)
    # Exact equality may fail at order-stat boundaries; allow small tolerance
    assert np.allclose(Rin[mask], R[mask], rtol=1e-6, atol=1e-7)

    # NaNs preserved after inverse
    assert np.isnan(Rin[3]) and np.isnan(Rin[7])


def test_nqt_min_maps_to_zero_and_sets_threshold():
    R = np.array([0.5, 0.7, 1.0, 0.2, 3.0])
    t = NQTransformer()
    Rt = t.transform(R)

    # The minimum non-NaN value should map to 0
    min_idx = np.nanargmin(R)
    assert np.isclose(Rt[min_idx], 0.0)

    # Threshold is the min positive transformed value
    positives = Rt[Rt > 0]
    assert np.isclose(t.metadata["threshold"], positives.min())


# ---------------------------
# SqrtTransformer
# ---------------------------


def test_sqrt_transform_and_inverse():
    t = SqrtTransformer(threshold=0.5)
    R = np.array([0.0, 0.25, 1.0, 4.0], dtype=float)
    Rt = t.transform(R)

    # Transform behavior
    assert np.allclose(Rt, np.sqrt(R))
    assert t.metadata["transform"] == "sqrt"
    assert np.isclose(t.metadata["threshold"], np.sqrt(0.5))
    assert np.isclose(t.metadata["zerovalue"], 0.0)  # default when not provided

    # Inverse behavior
    Rin = t.inverse_transform(Rt)
    assert np.allclose(Rin, R)
    assert t.metadata["transform"] is None


def test_sqrt_zerovalue_metadata_when_provided():
    t = SqrtTransformer(threshold=4.0, zerovalue=9.0)
    t.transform(np.array([0.0, 1.0]))
    assert np.isclose(t.metadata["zerovalue"], 3.0)  # sqrt(9.0)


# ---------------------------
# Factory
# ---------------------------


@pytest.mark.parametrize(
    "name, cls",
    [
        ("boxcox", BoxCoxTransformer),
        ("db", DBTransformer),
        ("nqt", NQTransformer),
        ("sqrt", SqrtTransformer),
        ("BoXcOx", BoxCoxTransformer),  # case-insensitive
    ],
)
def test_get_transformer_factory(name, cls):
    obj = get_transformer(name)
    assert isinstance(obj, cls)


def test_get_transformer_unknown_raises():
    with pytest.raises(ValueError):
        get_transformer("nope")
