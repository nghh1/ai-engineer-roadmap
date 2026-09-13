from src.validation import cross_validate_model, compare_models
from src.models import get_candidate_models
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pytest

@pytest.fixture
def data_and_models():
    models = get_candidate_models()
    X_train = pd.DataFrame({"age": [18, 22, 25, 30],
                     "monthly_spend": [300.5, 350.0, 350.4, 400.0],
                     "account_age_months": [10.5, 24.0, 32.2, 48.0],
                     "support_tickets": [100, 200, 300, 400],
                     "plan": [0, 1, 2, 1],
                     "country": [1, 2, 0, 2]})
    y_train = pd.Series([0, 1, 1, 0], name="churned")
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    return models, X_train, y_train, skf

def test_cross_validate_model_return(data_and_models):
    models, X_train, y_train, skf = data_and_models
    for model in models.values():
        cv_result = cross_validate_model(model, X_train, y_train, skf)
        assert cv_result is not None
        assert (cv_result["scores"] is not None) and (cv_result["mean_f1"] is not None) \
        and (cv_result["std_f1"] is not None)
        assert len(cv_result["scores"]) == skf.get_n_splits(X_train, y_train)
        assert (cv_result["mean_f1"] >= 0) & (cv_result["mean_f1"] <= 1)
        assert cv_result["std_f1"] >= 0

def test_compare_models_return(data_and_models):
    models, X_train, y_train, skf = data_and_models
    result_df = compare_models(models, X_train, y_train, skf)
    assert len(result_df) == len(models)
    assert all(column in result_df.columns for column in ["model", "mean_f1", "std_f1"])