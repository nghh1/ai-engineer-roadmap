import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def predict(model, X, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        logits = model(X).to(device)
        probs = torch.sigmoid(logits)
        predictions = (probs >= threshold).int()
    return probs, predictions

def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }