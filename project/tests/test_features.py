import numpy as np
import scipy.sparse as sp

from src.features.property_features import build_property_features


def test_feature_matrix_shape(properties, cfg):
    sample = properties.sample(n=500, random_state=1)
    fm, pids, artifacts = build_property_features(sample, cfg)
    assert fm.shape[0] == len(sample)
    assert len(pids) == len(sample)
    assert fm.shape[1] > 0


def test_feature_matrix_is_normalized(properties, cfg):
    sample = properties.sample(n=200, random_state=1)
    fm, pids, artifacts = build_property_features(sample, cfg)
    norms = sp.linalg.norm(fm, axis=1)
    nonzero = norms[norms > 0]
    assert np.allclose(nonzero, 1.0, atol=1e-6)


def test_feature_matrix_no_nans(properties, cfg):
    sample = properties.sample(n=200, random_state=1)
    fm, pids, artifacts = build_property_features(sample, cfg)
    assert not np.isnan(fm.data).any()
