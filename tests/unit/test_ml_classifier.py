"""
tests.unit.test_ml_classifier
================================

Comprehensive unit tests for:
  • phinance.strategies.ml_features  (build_features, build_labels)
  • phinance.strategies.ml_classifier (train_lgbm_model, LGBMClassifierIndicator)
  • Catalog integration for 'LGBM Classifier'

All tests use synthetic OHLCV data — no network calls, no live data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv

# ── Fixtures ───────────────────────────────────────────────────────────────────

DF_SMALL  = make_ohlcv(n=60,  start="2020-01-01")   # below default train_size=252
DF_MEDIUM = make_ohlcv(n=400, start="2021-01-01")   # above train_size, enough for rolling
DF_LARGE  = make_ohlcv(n=600, start="2022-01-01")   # ample data


# ═══════════════════════════════════════════════════════════════════════════════
# ml_features — build_features
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildFeatures:

    def test_returns_dataframe(self):
        from phinance.strategies.ml_features import build_features
        feat = build_features(DF_MEDIUM)
        assert isinstance(feat, pd.DataFrame)

    def test_same_index_as_input(self):
        from phinance.strategies.ml_features import build_features
        feat = build_features(DF_MEDIUM)
        assert feat.index.equals(DF_MEDIUM.index)

    def test_no_all_nan_columns(self):
        from phinance.strategies.ml_features import build_features
        feat = build_features(DF_MEDIUM)
        all_nan_cols = [c for c in feat.columns if feat[c].isna().all()]
        assert all_nan_cols == [], f"All-NaN columns: {all_nan_cols}"

    def test_feature_count_positive(self):
        from phinance.strategies.ml_features import build_features
        feat = build_features(DF_MEDIUM)
        assert feat.shape[1] > 0

    def test_finite_values_after_warmup(self):
        from phinance.strategies.ml_features import build_features
        feat = build_features(DF_MEDIUM).iloc[30:]  # skip warm-up
        assert np.isfinite(feat.values).all()

    def test_no_inf_values(self):
        from phinance.strategies.ml_features import build_features
        feat = build_features(DF_MEDIUM)
        assert not np.isinf(feat.values).any()

    def test_small_df_does_not_crash(self):
        from phinance.strategies.ml_features import build_features
        feat = build_features(DF_SMALL)
        assert isinstance(feat, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════════
# ml_features — build_labels
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildLabels:

    def test_returns_series(self):
        from phinance.strategies.ml_features import build_labels
        labels = build_labels(DF_MEDIUM)
        assert isinstance(labels, pd.Series)

    def test_same_length_as_input(self):
        from phinance.strategies.ml_features import build_labels
        labels = build_labels(DF_MEDIUM)
        assert len(labels) == len(DF_MEDIUM)

    def test_binary_values(self):
        from phinance.strategies.ml_features import build_labels
        labels = build_labels(DF_MEDIUM, horizon=1).dropna()
        unique = set(labels.unique())
        assert unique.issubset({0, 1})

    def test_last_horizon_bars_are_nan(self):
        from phinance.strategies.ml_features import build_labels
        horizon = 5
        labels = build_labels(DF_MEDIUM, horizon=horizon)
        # Last `horizon` rows should be NaN (no future return available)
        assert labels.iloc[-horizon:].isna().all()

    def test_horizon_1_default(self):
        from phinance.strategies.ml_features import build_labels
        labels = build_labels(DF_MEDIUM)
        non_null = labels.dropna()
        assert len(non_null) == len(DF_MEDIUM) - 1

    def test_threshold_affects_labels(self):
        from phinance.strategies.ml_features import build_labels
        lbl_zero    = build_labels(DF_MEDIUM, threshold=0.0).dropna()
        lbl_nonzero = build_labels(DF_MEDIUM, threshold=0.05).dropna()
        # With a positive threshold, fewer bars should be labelled 1
        assert lbl_nonzero.sum() <= lbl_zero.sum()


# ═══════════════════════════════════════════════════════════════════════════════
# train_lgbm_model
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainLGBMModel:

    def test_returns_fitted_model(self):
        from phinance.strategies.ml_classifier import train_lgbm_model
        model = train_lgbm_model(DF_LARGE, lgbm_params={"n_estimators": 10})
        assert model is not None

    def test_model_has_predict_proba(self):
        from phinance.strategies.ml_classifier import train_lgbm_model
        from phinance.strategies.ml_features import build_features
        model = train_lgbm_model(DF_LARGE, lgbm_params={"n_estimators": 10})
        feat = build_features(DF_LARGE)
        proba = model.predict_proba(feat)
        assert proba.shape == (len(DF_LARGE), 2)

    def test_proba_sums_to_one(self):
        from phinance.strategies.ml_classifier import train_lgbm_model
        from phinance.strategies.ml_features import build_features
        model = train_lgbm_model(DF_LARGE, lgbm_params={"n_estimators": 10})
        feat = build_features(DF_LARGE)
        proba = model.predict_proba(feat)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_custom_horizon(self):
        from phinance.strategies.ml_classifier import train_lgbm_model
        model = train_lgbm_model(DF_LARGE, horizon=3, lgbm_params={"n_estimators": 10})
        assert model is not None

    def test_model_save_load(self, tmp_path):
        from phinance.strategies.ml_classifier import train_lgbm_model, load_lgbm_model
        from phinance.strategies.ml_features import build_features
        path = str(tmp_path / "lgbm_test.pkl")
        model = train_lgbm_model(DF_LARGE, lgbm_params={"n_estimators": 5}, model_path=path)
        loaded = load_lgbm_model(path)
        feat = build_features(DF_LARGE)
        assert np.allclose(
            model.predict_proba(feat),
            loaded.predict_proba(feat),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LGBMClassifierIndicator
# ═══════════════════════════════════════════════════════════════════════════════

class TestLGBMClassifierIndicator:

    def test_instantiation(self):
        from phinance.strategies.ml_classifier import LGBMClassifierIndicator
        ind = LGBMClassifierIndicator()
        assert ind.name == "LGBM Classifier"

    def test_default_params_exist(self):
        from phinance.strategies.ml_classifier import LGBMClassifierIndicator
        ind = LGBMClassifierIndicator()
        assert "train_size" in ind.default_params
        assert "retrain_every" in ind.default_params

    def test_compute_returns_series(self):
        from phinance.strategies.ml_classifier import LGBMClassifierIndicator
        ind = LGBMClassifierIndicator(lgbm_params={"n_estimators": 10})
        sig = ind.compute(DF_LARGE, train_size=60, retrain_every=30)
        assert isinstance(sig, pd.Series)

    def test_signal_same_length(self):
        from phinance.strategies.ml_classifier import LGBMClassifierIndicator
        ind = LGBMClassifierIndicator(lgbm_params={"n_estimators": 10})
        sig = ind.compute(DF_LARGE, train_size=60, retrain_every=30)
        assert len(sig) == len(DF_LARGE)

    def test_signal_range(self):
        from phinance.strategies.ml_classifier import LGBMClassifierIndicator
        ind = LGBMClassifierIndicator(lgbm_params={"n_estimators": 10})
        sig = ind.compute(DF_LARGE, train_size=60, retrain_every=30)
        non_nan = sig.dropna()
        assert (non_nan >= -1.0).all() and (non_nan <= 1.0).all()

    def test_insufficient_data_returns_zeros(self):
        from phinance.strategies.ml_classifier import LGBMClassifierIndicator
        ind = LGBMClassifierIndicator()
        # DF_SMALL has 60 rows, default train_size=252 — insufficient
        sig = ind.compute(DF_SMALL, train_size=252)
        assert (sig == 0.0).all()

    def test_signal_name(self):
        from phinance.strategies.ml_classifier import LGBMClassifierIndicator
        ind = LGBMClassifierIndicator(lgbm_params={"n_estimators": 5})
        sig = ind.compute(DF_LARGE, train_size=60, retrain_every=30)
        assert sig.name == "LGBM Classifier"

    def test_compute_with_defaults_works(self):
        from phinance.strategies.ml_classifier import LGBMClassifierIndicator
        ind = LGBMClassifierIndicator(lgbm_params={"n_estimators": 5})
        # compute_with_defaults uses default_params
        sig = ind.compute_with_defaults(DF_LARGE)
        assert isinstance(sig, pd.Series)

    def test_no_nan_after_warm_up(self):
        from phinance.strategies.ml_classifier import LGBMClassifierIndicator
        ind = LGBMClassifierIndicator(lgbm_params={"n_estimators": 10})
        sig = ind.compute(DF_LARGE, train_size=60, retrain_every=200)
        # After training window, no NaN should exist
        post_warmup = sig.iloc[65:]
        assert not post_warmup.isna().any()

    def test_custom_lgbm_params_accepted(self):
        from phinance.strategies.ml_classifier import LGBMClassifierIndicator
        ind = LGBMClassifierIndicator(lgbm_params={
            "n_estimators": 5,
            "num_leaves": 7,
            "learning_rate": 0.1,
        })
        sig = ind.compute(DF_LARGE, train_size=60, retrain_every=30)
        assert isinstance(sig, pd.Series)


# ═══════════════════════════════════════════════════════════════════════════════
# Catalog integration for LGBM Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestLGBMCatalogIntegration:

    def test_lgbm_in_catalog(self):
        from phinance.strategies.indicator_catalog import list_indicators
        assert "LGBM Classifier" in list_indicators()

    def test_compute_indicator_lgbm(self):
        from phinance.strategies.indicator_catalog import compute_indicator
        sig = compute_indicator(
            "LGBM Classifier",
            DF_LARGE,
            {"train_size": 60, "retrain_every": 200},
        )
        assert isinstance(sig, pd.Series)
        assert len(sig) == len(DF_LARGE)

    def test_lgbm_signal_range_via_catalog(self):
        from phinance.strategies.indicator_catalog import compute_indicator
        sig = compute_indicator(
            "LGBM Classifier",
            DF_LARGE,
            {"train_size": 60, "retrain_every": 200},
        )
        non_nan = sig.dropna()
        assert (non_nan >= -1.0).all() and (non_nan <= 1.0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# param_grid for LGBM Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestLGBMParamGrid:

    def test_daily_grid_exists(self):
        from phinance.strategies.params import get_param_grid
        grid = get_param_grid("LGBM Classifier", "1D")
        assert isinstance(grid, dict)
        assert len(grid) > 0

    def test_intraday_grid_exists(self):
        from phinance.strategies.params import get_param_grid
        grid = get_param_grid("LGBM Classifier", "5m")
        assert isinstance(grid, dict)
        assert len(grid) > 0

    def test_daily_grid_has_expected_params(self):
        from phinance.strategies.params import get_param_grid
        grid = get_param_grid("LGBM Classifier", "1D")
        expected_keys = {"n_estimators", "num_leaves", "learning_rate", "lookback"}
        assert set(grid.keys()) == expected_keys

    def test_all_grid_values_are_lists(self):
        from phinance.strategies.params import get_param_grid
        grid = get_param_grid("LGBM Classifier", "1D")
        for key, vals in grid.items():
            assert isinstance(vals, list) and len(vals) > 0, \
                f"LGBM Classifier daily grid[{key}] is empty"
