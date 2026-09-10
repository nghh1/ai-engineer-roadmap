from src.preprocessing import split_features_target
from src.training import build_model
from sklearn.model_selection import train_test_split
import pytest
import pandas as pd

@pytest.fixture
def data_split():
    df = pd.DataFrame({
        "customer_id": [1000, 1001, 1002, 1003],
        "age": [18, 19, 20, 21], 
        "monthly_spend": [100, 110, 120, 130], 
        "account_age_months": [40, 40.5, 45.5, 50], 
        "support_tickets": [30, 45, 40, 50],
        "plan": ["basic", "basic", "standard", "premium"], 
        "country": ["uk", "spain", "france", "germany"],
        "churned": [1, 0, 1, 0]
    })
    X, y = split_features_target(df)
    return X, y

def test_churned_removed(data_split):
    X, _ = data_split
    assert ("churned" not in X.columns)

def test_customerid_removed(data_split):
    X, _ = data_split
    assert ("customer_id" not in X.columns)

def test_y_contains_churned(data_split):
    _, y = data_split
    assert ("churned" in y.name)

def test_model_success(data_split):
    model = build_model()
    X, y = data_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred is not None
    assert len(y_pred) == len(y_test)