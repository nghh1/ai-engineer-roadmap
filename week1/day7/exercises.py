import sys
from pathlib import Path
week1_dir = Path(__file__).resolve().parent.parent
mini_project_dir = week1_dir / "day6" / "mini_project"
sys.path.extend([str(week1_dir), str(mini_project_dir)])

from day6.mini_project.src.models import build_decision_tree
# Exercise 1: inspect pipeline parameters()
model = build_decision_tree()
params = model.get_params()
print(params)
"""
'dt__max_depth': 3, 'dt__min_samples_leaf': 1, 'dt__min_samples_split': 2
dt__ prefix indicates the decisiontree classifier step in the Pipeline object.
"""

# Exercise 2: tune decision tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from day5.mini_project.src.loading import load_data
from day5.mini_project.src.preprocessing import split_features_target
param_grid = {
    "dt__max_depth": [2, 3, 4, 5, None],
    "dt__min_samples_leaf": [1, 2, 4],
    "dt__min_samples_split": [2, 5, 10]
}
df = load_data("../day6/mini_project/data/customer.csv")
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(model, param_grid, scoring="f1", cv=skf)
search.fit(X_train, y_train)
print(f"\n{search.best_params_}")
print(search.best_score_)
"""
{'dt__max_depth': 5, 'dt__min_samples_leaf': 4, 'dt__min_samples_split': 10}
0.5879365079365079
"""

# Exercise 3: inspect search results
import pandas as pd
result_df = pd.DataFrame(search.cv_results_)
selected = result_df[["param_dt__max_depth", "param_dt__min_samples_split", 
           "param_dt__min_samples_leaf", "mean_test_score", "std_test_score", 
           "rank_test_score"]]
sorted_selected = selected.sort_values(by="rank_test_score").head(5)
print(sorted_selected)
"""
   param_dt__max_depth  param_dt__min_samples_split  \
35                   5                           10   
44                None                           10   
21                   4                            2   
22                   4                            5   
26                   4                           10   

    param_dt__min_samples_leaf  mean_test_score  std_test_score  \
35                           4         0.587937        0.139501   
44                           4         0.584495        0.154523   
21                           2         0.579802        0.087634   
22                           2         0.579802        0.087634   
26                           4         0.573200        0.141328   

    rank_test_score  
35                1  
44                2  
21                3  
22                3  
26                5
The top configurations are somewhat different, but several favour stronger regularisation 
through larger min_samples_leaf or min_samples_split. Their CV scores are very close, 
suggesting there isn't one overwhelmingly superior configuration.
"""

# Exercise 4: final holdout evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
best_tree = search.best_estimator_
y_pred = best_tree.predict(X_test)
y_prob = best_tree.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"accuracy: {accuracy}, precision: {precision}")
print(f"recall: {recall}, f1: {f1}")
print(f"roc_auc: {roc_auc}")
"""
accuracy: 0.625, precision: 0.5384615384615384
recall: 0.5384615384615384, f1: 0.5384615384615384
roc_auc: 0.6194331983805668
"""

# Exercise 5: tune logistic regression
from day6.mini_project.src.models import build_logistic_regression
model = build_logistic_regression()
param_grid = {"LR__C": [0.01, 0.1, 1, 10, 100]} # smaller C stronger regularisation
search_lr = GridSearchCV(model, param_grid, scoring='f1', cv=skf)
search_lr.fit(X_train, y_train)
print(search_lr.best_params_)
print(search_lr.best_score_)
"""
{'LR__C': 1}
0.6018713450292397
"""

# Exercise 6: RandomizedSearchCV with Random Forest
from sklearn.model_selection import RandomizedSearchCV
from day6.mini_project.src.models import build_random_forest
model = build_random_forest()
param_distributions = {
    "rf__n_estimators": [50, 100, 200, 300],
    "rf__max_depth": [None, 3, 5, 8, 12],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2", None],
}
search_rf = RandomizedSearchCV(model, param_distributions, n_iter=20, scoring="f1", cv=skf, random_state=42)
search_rf.fit(X_train, y_train)
print(search_rf.best_params_)
print(search_rf.best_score_)
"""
{'rf__n_estimators': 200, 'rf__min_samples_split': 10, 'rf__min_samples_leaf': 2, 
'rf__max_features': None, 'rf__max_depth': 8}

0.6789746201510908
"""

# Exercise 7: baseline vs tuned models
from sklearn.model_selection import cross_val_score
baseline_LR = build_logistic_regression()
baseline_DT = build_decision_tree()
baseline_RF = build_random_forest()
baseline_LR_scores = cross_val_score(baseline_LR, X_train, y_train, scoring='f1', cv=skf)
baseline_DT_scores = cross_val_score(baseline_DT, X_train, y_train, scoring='f1', cv=skf)
baseline_RF_scores = cross_val_score(baseline_RF, X_train, y_train, scoring='f1', cv=skf)

baseline_vs_tuned = pd.DataFrame({"model": ["Logistic Regression", "Decision Tree", "Random Forest"], 
                                  "baseline CV F1": [baseline_LR_scores.mean(), baseline_DT_scores.mean(), baseline_RF_scores.mean()], 
                                  "tuned CV F1": [search_lr.best_score_, search.best_score_, search_rf.best_score_]})
print(baseline_vs_tuned)
"""
                 model  baseline CV F1  tuned CV F1
0  Logistic Regression        0.601871     0.601871
1        Decision Tree        0.556176     0.587937
2        Random Forest        0.637303     0.678975
Tuning improved Decision Tree and Random Forest, but did not improve Logistic Regression 
because its default C=1 was already the best value in the tested search space.
"""