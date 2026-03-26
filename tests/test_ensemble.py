"""Tests for EnsembleForecaster: weighted averaging, model combination."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import EnsembleForecaster
from src.models.statistical import BaseForecaster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubForecaster(BaseForecaster):
    """Stub forecaster that returns predetermined values for testing."""

    def __init__(self, name: str, values: list[float]) -> None:
        super().__init__(name=name)
        self._values = values
        self.is_fitted = True

    def fit(self, y: pd.Series, **kwargs: Any) -> None:
        self.is_fitted = True

    def predict(self, horizon: int, **kwargs: Any) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        arr = self._values[:horizon]
        if len(arr) < horizon:
            arr = arr + [arr[-1]] * (horizon - len(arr))
        return pd.Series(arr, name=f"{self.name}_forecast")


class FailingForecaster(BaseForecaster):
    """Forecaster that always raises on predict."""

    def __init__(self) -> None:
        super().__init__(name="Failing")
        self.is_fitted = True

    def fit(self, y: pd.Series, **kwargs: Any) -> None:
        self.is_fitted = True

    def predict(self, horizon: int, **kwargs: Any) -> pd.Series:
        raise RuntimeError("Prediction failed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_a() -> StubForecaster:
    return StubForecaster("ModelA", [100.0, 110.0, 120.0, 130.0, 140.0])


@pytest.fixture
def model_b() -> StubForecaster:
    return StubForecaster("ModelB", [200.0, 210.0, 220.0, 230.0, 240.0])


@pytest.fixture
def model_c() -> StubForecaster:
    return StubForecaster("ModelC", [150.0, 160.0, 170.0, 180.0, 190.0])


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestEnsembleInit:
    """Tests for EnsembleForecaster initialization."""

    def test_default_init(self) -> None:
        ensemble = EnsembleForecaster()
        assert ensemble.models == []
        assert ensemble.method == "weighted_average"
        assert ensemble.weights == {}

    def test_init_with_models(self, model_a: StubForecaster, model_b: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a, model_b])
        assert len(ensemble.models) == 2

    def test_add_model(self, model_a: StubForecaster) -> None:
        ensemble = EnsembleForecaster()
        ensemble.add_model(model_a)
        assert len(ensemble.models) == 1
        assert ensemble.models[0].name == "ModelA"


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

class TestWeightComputation:
    """Tests for backtest-based weight computation."""

    def test_equal_weights_no_scores(self, model_a: StubForecaster, model_b: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a, model_b])
        ensemble._compute_weights()
        assert abs(ensemble.weights["ModelA"] - 0.5) < 1e-6
        assert abs(ensemble.weights["ModelB"] - 0.5) < 1e-6

    def test_inverse_error_weighting(self, model_a: StubForecaster, model_b: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a, model_b])
        # ModelA has lower error -> should get higher weight
        ensemble.set_backtest_scores({"ModelA": 10.0, "ModelB": 30.0})
        assert ensemble.weights["ModelA"] > ensemble.weights["ModelB"]

    def test_weights_sum_to_one(self, model_a: StubForecaster, model_b: StubForecaster, model_c: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a, model_b, model_c])
        ensemble.set_backtest_scores({"ModelA": 15.0, "ModelB": 25.0, "ModelC": 10.0})
        total = sum(ensemble.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_zero_score_gets_default_weight(self, model_a: StubForecaster, model_b: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a, model_b])
        ensemble.set_backtest_scores({"ModelA": 10.0, "ModelB": 0.0})
        # ModelB with zero score should get default weight of 1.0
        assert ensemble.weights["ModelB"] > 0


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class TestEnsemblePrediction:
    """Tests for ensemble prediction methods."""

    def test_simple_average(self, model_a: StubForecaster, model_b: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a, model_b], method="simple_average")
        result = ensemble.predict(horizon=3)
        expected = [(100 + 200) / 2, (110 + 210) / 2, (120 + 220) / 2]
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_weighted_average(self, model_a: StubForecaster, model_b: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a, model_b], method="weighted_average")
        # ModelA error=10, ModelB error=40 -> ModelA gets 4x weight
        ensemble.set_backtest_scores({"ModelA": 10.0, "ModelB": 40.0})
        result = ensemble.predict(horizon=1)
        # weight_A = 1/10 = 0.1, weight_B = 1/40 = 0.025
        # normalized: A=0.8, B=0.2
        expected = 0.8 * 100.0 + 0.2 * 200.0
        assert abs(result.iloc[0] - expected) < 1e-6

    def test_best_method(self, model_a: StubForecaster, model_b: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a, model_b], method="best")
        ensemble.set_backtest_scores({"ModelA": 10.0, "ModelB": 30.0})
        result = ensemble.predict(horizon=3)
        # Should use only ModelA (lowest error)
        np.testing.assert_array_almost_equal(result.values, [100.0, 110.0, 120.0])

    def test_predict_returns_series(self, model_a: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a])
        result = ensemble.predict(horizon=3)
        assert isinstance(result, pd.Series)
        assert result.name == "ensemble_forecast"

    def test_predict_correct_horizon(self, model_a: StubForecaster, model_b: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a, model_b])
        result = ensemble.predict(horizon=5)
        assert len(result) == 5

    def test_no_models_raises(self) -> None:
        ensemble = EnsembleForecaster(models=[])
        with pytest.raises(RuntimeError, match="no fitted models"):
            ensemble.predict(horizon=5)

    def test_unfitted_models_excluded(self, model_a: StubForecaster) -> None:
        unfitted = StubForecaster("Unfitted", [999.0])
        unfitted.is_fitted = False
        ensemble = EnsembleForecaster(models=[model_a, unfitted])
        result = ensemble.predict(horizon=3)
        # Should only use model_a
        np.testing.assert_array_almost_equal(result.values, [100.0, 110.0, 120.0])

    def test_failing_model_gracefully_handled(self, model_a: StubForecaster) -> None:
        failing = FailingForecaster()
        ensemble = EnsembleForecaster(models=[model_a, failing])
        result = ensemble.predict(horizon=3)
        # Should fall back to model_a only
        np.testing.assert_array_almost_equal(result.values, [100.0, 110.0, 120.0])

    def test_all_models_fail_raises(self) -> None:
        failing = FailingForecaster()
        ensemble = EnsembleForecaster(models=[failing])
        with pytest.raises(RuntimeError, match="All models failed"):
            ensemble.predict(horizon=3)


# ---------------------------------------------------------------------------
# Individual forecasts and summary
# ---------------------------------------------------------------------------

class TestEnsembleUtilities:
    """Tests for get_individual_forecasts and summary."""

    def test_individual_forecasts_dataframe(self, model_a: StubForecaster, model_b: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a, model_b])
        df = ensemble.get_individual_forecasts(horizon=3)
        assert isinstance(df, pd.DataFrame)
        assert "ModelA" in df.columns
        assert "ModelB" in df.columns
        assert "ensemble" in df.columns
        assert len(df) == 3

    def test_summary_structure(self, model_a: StubForecaster, model_b: StubForecaster) -> None:
        ensemble = EnsembleForecaster(models=[model_a, model_b])
        ensemble.set_backtest_scores({"ModelA": 10.0, "ModelB": 20.0})
        summary = ensemble.summary()
        assert summary["method"] == "weighted_average"
        assert summary["n_models"] == 2
        assert "ModelA" in summary["models"]
        assert "ModelB" in summary["models"]
        assert len(summary["weights"]) == 2
        assert summary["backtest_scores"]["ModelA"] == 10.0
