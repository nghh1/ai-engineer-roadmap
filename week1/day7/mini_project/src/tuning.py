from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
LOGISTIC_GRID = {
    "LR__C": [0.01, 0.1, 1, 10, 100]
}
TREE_GRID = {
    "DT__max_depth": [2, 3, 4, 5, None],
    "DT__min_samples_split": [2, 5, 10],
    "DT__min_samples_leaf": [1, 2, 4]
}
RF_DISTRIBUTIONS = {
    "RF__n_estimators": [50, 100, 200, 300],
    "RF__max_depth": [None, 3, 5, 8, 12],
    "RF__min_samples_split": [2, 5, 10],
    "RF__min_samples_leaf": [1, 2, 4],
    "RF__max_features": ["sqrt", "log2", None],
}

def tune_with_grid(model, param_grid, X, y, cv):
    search = GridSearchCV(model, param_grid, scoring="f1", cv=cv)
    search.fit(X, y)
    return search

def tune_with_random_search(model, param_distributions, X, y, cv):
    search = RandomizedSearchCV(model, param_distributions, n_iter=20, scoring='f1', cv=cv, random_state=42)
    search.fit(X, y)
    return search

def tune_all_models(models, X_train, y_train, cv):
    df = pd.DataFrame()
    searches = {}
    rows = []
    for name, model in models.items():
        if name =="logistic_regression":
            searches[name] = tune_with_grid(model, LOGISTIC_GRID, X_train, y_train, cv)
        elif name=="decision_tree":
            searches[name] = tune_with_grid(model, TREE_GRID, X_train, y_train, cv)
        else:
            searches[name] = tune_with_random_search(model, RF_DISTRIBUTIONS, X_train, y_train, cv)
        rows.append({"model": name, "best_cv_f1": searches[name].best_score_, "best_params": searches[name].best_params_})
        df = pd.DataFrame(rows)
    df = df.sort_values(by='best_cv_f1', ascending=False, ignore_index=True)
    return df, searches

def best_model_return(df, searches):
    best_model_name = df.iloc[0]["model"]
    best_model = searches[best_model_name].best_estimator_
    return best_model


        