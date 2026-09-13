from src.models import get_candidate_models
from sklearn.exceptions import NotFittedError
import pandas as pd

def test_model_pipeline():
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
    for model in models.values():
        model.fit(X, y)
        y_pred = model.predict(X)
        assert len(y_pred) == len(X)
        assert all(x in {0, 1} for x in y_pred)
        