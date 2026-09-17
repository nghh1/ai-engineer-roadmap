from src.evaluation import evaluate_model

def test_evaluation():
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    y_prob = [0.6, 0.5, 0.3, 0.8]
    results = evaluate_model(y_true, y_pred, y_prob)
    assert results == {"accuracy": 0.75, "precision": 1.0, "recall": 2/3, "f1": 0.8, "roc_auc": 2/3}
