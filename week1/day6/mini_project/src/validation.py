from sklearn.model_selection import cross_val_score
import pandas as pd

def cross_validate_model(model, X, y, cv) -> dict:
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    return {
        "scores": scores,
        "mean_f1": scores.mean(),
        "std_f1": scores.std()
    }

def compare_models(models: dict, X, y, cv) -> pd.DataFrame:
    rows = []
    df = pd.DataFrame({"model": list(), "mean_f1": list(), "std_f1": list()})
    for name, model in models.items():
        scores = cross_validate_model(model, X, y, cv)
        rows.append({"model": name, "mean_f1": scores["mean_f1"], "std_f1": scores["std_f1"]})
        df = pd.DataFrame(rows)
    df.sort_values(by=["mean_f1"], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df