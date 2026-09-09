import pandas as pd
from src.cleaning import clean_data

def test_model_name_normalised():
    df = pd.DataFrame({
        "request_id": [1, 2, 3],
        "model": ["  GPT ", "gpt", "Claude"],
        "prompt_tokens": [100, 100, 130],
        "completion_tokens": [80, 80, 110],
        "latency_ms": [100, 100, 100]
    })
    cleaned = clean_data(df)
    pd.testing.assert_series_equal(cleaned["model"], pd.Series(["gpt", "gpt", "claude"], name="model"))

def test_clean_data_not_mutate_input():
    df = pd.DataFrame({
        "request_id": [1, 2, 3],
        "model": ["gpt", "gpt", "claude"],
        "prompt_tokens": [100, 100, 130],
        "completion_tokens": [80, 90, 110],
        "latency_ms": [100, 100, 100]
    })
    original = df.copy()
    clean_data(df)
    pd.testing.assert_frame_equal(original, df)

def test_no_duplicated_rows():
    df = pd.DataFrame({
        "request_id": [1, 1, 2],
        "model": ["gpt ", "gpt", "Claude"],
        "prompt_tokens": [100, 100, 130],
        "completion_tokens": [80, 80, 110],
        "latency_ms": [100, 100, 100]
    })
    cleaned = clean_data(df)
    assert cleaned.duplicated().sum() == 0

def test_numeric_na_filled():
    df = pd.DataFrame({
        "request_id": [1, 2, 3, 4],
        "model": ["gpt", "gpt", "claude", "gemini"],
        "prompt_tokens": [100, 130, "unknown", 140],
        "completion_tokens": [80, 110, 140, 120],
        "latency_ms": [100, 100, 100, 100]
    })
    cleaned = clean_data(df)
    assert pd.isna(cleaned["prompt_tokens"]).sum() == 0