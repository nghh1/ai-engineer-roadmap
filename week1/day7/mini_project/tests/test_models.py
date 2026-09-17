import pytest
import pandas as pd
from src.models import get_candidate_models
from sklearn.pipeline import Pipeline

@pytest.fixture
def data_and_models():
    X = pd.DataFrame({
        "age": [18, 22, 25, 30, 35, 42],
        "monthly_spend": [300.5, 350.0, 280.0, 400.0, 250.0, 500.0],
        "account_age_months": [10.5, 24.0, 32.2, 48.0, 15.0, 60.0],
        "support_tickets": [1, 3, 2, 5, 0, 4],
        "plan": ["basic", "standard", "premium", "basic", "standard", "premium"],
        "country": ["uk", "france", "germany", "uk", "spain", "france"]
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    models = get_candidate_models()
    return X, y, models

def test_model_pipeline(data_and_models):
    X, y, models = data_and_models
    for m in models.values():
        assert type(m) == Pipeline
        m.fit(X, y)
        y_pred = m.predict(X)
        assert len(y_pred) == len(y)
        assert all(p in {0, 1} for p in y_pred)

