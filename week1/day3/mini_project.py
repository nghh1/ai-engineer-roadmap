import numpy as np
y_true = np.array([
    1, 0, 1, 1, 0,
    1, 0, 0, 1, 0
])

y_prob = np.array([
    0.92, 0.31, 0.81, 0.44, 0.18,
    0.67, 0.55, 0.22, 0.73, 0.40
])

def predict_labels(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold)

def confusion_counts(y_true, y_pred):
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return TP, TN, FP, FN

def accuracy(y_true, y_pred) -> float:
    TP, TN, FP, FN = confusion_counts(y_true, y_pred)
    if (TP + TN + FP + FN) == 0:
        return 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

def precision(y_true, y_pred) -> float:
    TP, _, FP, _ = confusion_counts(y_true, y_pred)
    if (TP + FP) ==0:
        return 0.0
    precision = TP / (TP + FP)
    return precision
        
def recall(y_true, y_pred) -> float:
    TP, _, _, FN = confusion_counts(y_true, y_pred)
    if (TP + FN) == 0:
        return 0.0 
    recall = TP / (TP + FN)
    return recall

def f1_score(precision: float, recall: float) -> float:
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

if __name__ == "__main__":
    thresholds = np.array([0.3, 0.5, 0.7])
    for t in thresholds:
        y_pred = predict_labels(y_prob, t)
        a = accuracy(y_true, y_pred)
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        f1 = f1_score(p, r)
        print(f"Threshold {t}: accuracy = {a}, precision = {p}, recall = {r}, f1 score = {f1}")