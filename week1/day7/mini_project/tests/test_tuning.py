from src.tuning import tune_all_models, best_model_return
from src.models import get_candidate_models
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import pandas as pd
import pytest

@pytest.fixture
def data_and_models():
    X_train = pd.DataFrame({"age": [18, 22, 25, 30],
                     "monthly_spend": [300.5, 350.0, 350.4, 400.0],
                     "account_age_months": [10.5, 24.0, 32.2, 48.0],
                     "support_tickets": [100, 200, 300, 400],
                     "plan": [0, 1, 2, 1],
                     "country": [1, 2, 0, 2]})
    y_train = pd.Series([0, 1, 1, 0], name="churned")
    models = get_candidate_models()
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    return models, skf, X_train, y_train

def test_fitted_search(data_and_models):
    models, skf, X_train, y_train = data_and_models
    compare_df, searches = tune_all_models(models, X_train, y_train, cv=skf)
    for search in searches.values():
        check_is_fitted(search)
        assert all(0<=score<=1 for score in compare_df["best_cv_f1"])
        assert "best_params" in compare_df.columns
        assert best_model_return(compare_df, searches) is not None
        assert len(compare_df) == len(models)
        assert compare_df["best_cv_f1"].is_monotonic_decreasing
        assert compare_df.iloc[0]["best_cv_f1"] == compare_df["best_cv_f1"].max()

