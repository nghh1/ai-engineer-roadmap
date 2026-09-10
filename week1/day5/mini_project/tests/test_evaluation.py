from src.evaluation import evaluate_model
from pytest import approx

def test_model_evaluation():
    y_test = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    result = evaluate_model(y_test, y_pred)
    expected = {
        "accuracy": 0.75,
        "precision": 1.0,
        "recall": approx(0.667, rel=1e-3),
        "f1": 0.80
    }
    assert result == expected
