from src.loading import load_data
from src.preprocessing import split_features_target
from sklearn.model_selection import train_test_split, StratifiedKFold
from src.models import get_candidate_models
from src.tuning import tune_all_models, best_model_return
from src.evaluation import evaluate_model

def main(path):
    df = load_data(path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models  = get_candidate_models()
    df, searches = tune_all_models(models, X_train, y_train, cv)
    print(df)
    best_model = best_model_return(df, searches)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    results = evaluate_model(y_test, y_pred, y_prob)
    print(results)

if __name__=="__main__":
    path = "./data/customer.csv"
    main(path)
