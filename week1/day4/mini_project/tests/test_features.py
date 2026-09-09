import pandas as pd
import numpy as np
from src.features import add_features
from pytest import approx
import pytest

@pytest.fixture
def adjusted_df():
    df = pd.DataFrame({
        "request_id": [1, 2, 3, 4],
        "model": ["gpt", "grok", "claude", "gemini"],
        "prompt_tokens": [100, 130, 140, 120],
        "completion_tokens": [80, 110, 140, 120],
        "latency_ms": [1100, 0, 1200, 1000]
    })
    return add_features(df)

def test_total_tokens(adjusted_df):
    pd.testing.assert_series_equal(adjusted_df["total_tokens"], pd.Series([180, 240, 280, 240], name="total_tokens"))

def test_latency_seconds(adjusted_df):
    pd.testing.assert_series_equal(adjusted_df["latency_seconds"], pd.Series([1.1, 0.0, 1.2, 1.0], name="latency_seconds"))

def test_zero_latency(adjusted_df):
    pd.testing.assert_series_equal(pd.isna(adjusted_df["tokens_per_second"]), pd.Series([False, True, False, False], name="tokens_per_second"))

def test_token_per_second(adjusted_df):
    assert adjusted_df["tokens_per_second"].to_numpy() == approx(pd.Series([163.64, np.nan, 233.33, 240.00], name="tokens_per_second"), rel=1e-2, nan_ok=True)
