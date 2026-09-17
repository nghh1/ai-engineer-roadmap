# GridSearchCV tries a grid of configurations using cross-validation
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

model = Pipeline([('preprocessor', build_preprocessor()), \
                  ('classifier', DecisionTreeClassifier())])
param_grid = {
    "classifier__max_depth": [3, 5, 10, None],
    "classifier__min_samples_split": [2, 5, 10]
}
search = GridSearchCV(estimator=model, param_grid=param_grid, scoring="f1", cv=5)
search.fit(X_train, y_train)
## Returns best parameter setting, its score, and the model with best configuration.
search.best_params_
search.best_score_
search.best_estimator_
## Output a list of hyperparameters that can be tuned
model.get_params().keys()
## Returns: classifier, classifier__max_depth, classifier__min_samples_split, preprocessor

## Inside GridSearchCV, with param_grid and num_splits
## Conceptually the flow of validating max_depth: max_depth -> k-fold CV -> scores -> mean scores. Repeated for n configs and compare them to get best configuration.

## Today's focus on X_train / y_train -> GridSearchCV -> CV -> best hyperparameters -> best estimator before evaluating on X_test / y_test.
After search.fit() get search.best_estimator_, then best_model.predict() gets y_pred.

## search.cv_results_ to demonstrate search results in detail
result = pd.DataFrame(search.cv_results_)
result[["params", "mean_test_score", "std_test_score", "rank_test_score"]]

# RandomizedSearchCV samples a fixed number of configurations
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model, params, n_iter=20, scoring="f1", cv=5, random_state=42)