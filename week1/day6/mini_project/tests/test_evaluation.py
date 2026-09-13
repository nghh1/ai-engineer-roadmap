from src.evaluation import evaluate_model

def test_evaluate_model():
    y_test = [1, 1, 0, 1]
    y_pred = [1, 0, 0, 1]
    y_prob = [0.6, 0.3, 0.5, 0.8]
    result = evaluate_model(y_test, y_pred, y_prob)
    expected = {
        "accuracy": 0.75,
        "precision": 1.0,
        "recall": 2/3,
        "f1": 0.8,
        "roc_auc": 2/3
    }
    assert result == expected

