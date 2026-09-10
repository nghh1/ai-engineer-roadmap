from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_test, y_pred) -> dict:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def get_confusion_matrix(y_test, y_pred):
    TP = ((y_test==0) & (y_pred==0)).sum()
    TN = ((y_test==1) & (y_pred==1)).sum()
    FP = ((y_test==1) & (y_pred==0)).sum()
    FN = ((y_test==0) & (y_pred==1)).sum()
    return TP, TN, FP, FN