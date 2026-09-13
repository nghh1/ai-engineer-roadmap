from src.loading import loading_data
from src.preprocessing import split_features_target
from sklearn.model_selection import train_test_split, StratifiedKFold
from src.models import get_candidate_models
from src.validation import compare_models
from src.evaluation import evaluate_model

def main(path):
    df = loading_data(path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_candidate_models()
    compare_result = compare_models(models, X_train, y_train, skf)
    best_model_name = compare_result.iloc[0]["model"]
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred, y_prob = best_model.predict(X_test), best_model.predict_proba(X_test)[:, 1]
    holdout_result = evaluate_model(y_test, y_pred, y_prob)
    print(holdout_result)

if __name__ == "__main__":
    path = "data/customer.csv"
    main(path)