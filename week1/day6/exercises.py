import sys
from pathlib import Path

week1_dir = Path(__file__).resolve().parent.parent
mini_project_dir = week1_dir / "day5" / "mini_project"

sys.path.extend([str(week1_dir), str(mini_project_dir)])

from day5.mini_project.src.loading import load_data
from day5.mini_project.src.preprocessing import split_features_target
from sklearn.model_selection import train_test_split
from day5.mini_project.src.training import build_model
from sklearn.metrics import accuracy_score, f1_score
# Exercise 1: train vs test performance
df = load_data(f"{mini_project_dir}/data/customer.csv")
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = build_model()
model = model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
acc_y_pred_train = accuracy_score(y_train, y_pred_train)
f1_y_pred_train = f1_score(y_train, y_pred_train)
y_pred_test = model.predict(X_test)
acc_y_pred_test = accuracy_score(y_test, y_pred_test)
f1_y_pred_test = f1_score(y_test, y_pred_test)
print(f"train accuracy: {acc_y_pred_train}, train f1: {f1_y_pred_train}")
print(f"test accuracy: {acc_y_pred_test}, test f1: {f1_y_pred_test}")
"""
train accuracy: 0.7578125, train f1: 0.6666666666666666
test accuracy: 0.78125, test f1: 0.6956521739130435
No evidence showed that it was an overfitting, it looks underfitting instead.
"""

# Exercise 2: 5-fold cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
print(f"{scores}, {scores.mean()}, {scores.std()}")
## [0.42857143 0.73684211 0.4 0.5 0.72727273], 0.5585372522214627, 0.14540706084341232
"""
K cross-validation mean F1 can be more trustworthy since it was from different validation proportions, instead of one proportion only.
"""

# Exercise 3: Stratified KFold
import pandas as pd
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# skf.split() gives the rows indexes
for train_idx, validation_idx in skf.split(X_train, y_train):
    y_train_fold, y_validatiom_fold = y_train.iloc[train_idx], y_train.iloc[validation_idx]
    print(f"Train: {pd.Series(y_train_fold).value_counts()}"), 
    print(f"Validation: {pd.Series(y_validatiom_fold).value_counts()}")
"""
Train: churned
0    61
1    41
Name: count, dtype: int64
Validation: churned
0    16
1    10
Name: count, dtype: int64
Train: churned
0    61
1    41
Name: count, dtype: int64
Validation: churned
0    16
1    10
Name: count, dtype: int64
Train: churned
0    62
1    40
Name: count, dtype: int64
Validation: churned
0    15
1    11
Name: count, dtype: int64
Train: churned
0    62
1    41
Name: count, dtype: int64
Validation: churned
0    15
1    10
Name: count, dtype: int64
Train: churned
0    62
1    41
Name: count, dtype: int64
Validation: churned
0    15
1    10
Name: count, dtype: int64
"""

# Exercise 4: compare metrics
from day5.mini_project.src.evaluation import evaluate_model
from sklearn.metrics import roc_auc_score
y_pred_test = model.predict(X_test)
result = evaluate_model(y_test, y_pred_test)
print(f"test accuracy: {result["accuracy"]}, test precision: {result["precision"]}")
print(f"test recall: {result["recall"]}, test f1: {result["f1"]}")
y_prob_test = model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_prob_test)
print(f"ROC-AUC: {roc_auc}")
"""
test accuracy: 0.78125, test precision: 0.8
test recall: 0.6153846153846154, test f1: 0.6956521739130435
ROC-AUC: 0.8421052631578948
"""

# Exercise 5: intentionally overfit
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from day5.mini_project.src.preprocessing import build_preprocessor
dtc_unrestricted =  Pipeline([("preprocess", build_preprocessor()), ("dte", DecisionTreeClassifier(random_state=42))])
dtc_unrestricted = dtc_unrestricted.fit(X_train, y_train)
y_pred_train = dtc_unrestricted.predict(X_train)
y_pred_test = dtc_unrestricted.predict(X_test)
f1_train_dtc_unrestricted = f1_score(y_train, y_pred_train)
f1_test_dtc_unrestricted = f1_score(y_test, y_pred_test)
print(f"Unrestricted train f1: {f1_train_dtc_unrestricted}, unrestricted test f1: {f1_test_dtc_unrestricted}")
dtc_restricted = Pipeline([("preprocess", build_preprocessor()), ("dte", DecisionTreeClassifier(max_depth=3, random_state=42))])
dtc_restricted = dtc_restricted.fit(X_train, y_train)
y_pred_train = dtc_restricted.predict(X_train)
y_pred_test = dtc_restricted.predict(X_test)
f1_train_dtc_restricted = f1_score(y_train, y_pred_train)
f1_test_dtc_restricted = f1_score(y_test, y_pred_test)
print(f"Restricted train f1: {f1_train_dtc_restricted}, restricted test f1: {f1_test_dtc_restricted}")
"""
Unrestricted train f1: 1.0, unrestricted test f1: 0.5161290322580645
Restricted train f1: 0.7692307692307693, restricted test f1: 0.6
"""

# Exercise 6: model comparison
scores_LR = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
scores_dtc_restricted = cross_val_score(dtc_restricted, X_train, y_train, cv=5, scoring="f1")
scores_dtc_unrestricted = cross_val_score(dtc_unrestricted, X_train, y_train, cv=5, scoring="f1")
print(f"LR CV mean F1: {scores_LR.mean()}, LR CV std: {scores_LR.std()}")
print(f"dtc_restricted CV mean F1: {scores_dtc_restricted.mean()}, dtc_restricted CV std: {scores_dtc_restricted.std()}")
print(f"dtc_unrestricted CV mean F1: {scores_dtc_unrestricted.mean()}, dtc_unrestricted CV std: {scores_dtc_unrestricted.std()}")
"""
LR CV mean F1: 0.5585372522214627, LR CV std: 0.14540706084341232
dtc_restricted CV mean F1: 0.3946172248803827, dtc_restricted CV std: 0.12695237049736346
dtc_unrestricted CV mean F1: 0.5176577808156756, dtc_unrestricted CV std: 0.09174325538050818
"""

# Exercise 7: threshold revisit
from sklearn.metrics import precision_score, recall_score
import numpy as np
y_prob = model.predict_proba(X_test)[:, 1]
threshold_df = pd.DataFrame({'threshold': list(),
                             'precision': list(),
                             'recall': list(),
                             'f1': list()
                             })
for i, threshold in enumerate(np.arange(0.3, 0.8, step=0.1)):
    y_prob_threshold = (y_prob >= threshold).astype(int)
    precision = precision_score(y_test, y_prob_threshold)
    recall = recall_score(y_test, y_prob_threshold)
    f1 = f1_score(y_test, y_prob_threshold)
    new_row = {'threshold': threshold, 'precision': precision, 'recall': recall, 'f1': f1}
    threshold_df = pd.concat([threshold_df, pd.DataFrame([new_row])], ignore_index=True)
print(threshold_df)
"""
   threshold  precision    recall        f1
0        0.3   0.631579  0.923077  0.750000
1        0.4   0.666667  0.769231  0.714286
2        0.5   0.800000  0.615385  0.695652
3        0.6   0.857143  0.461538  0.600000
4        0.7   0.800000  0.307692  0.444444
Threshold = 0.3 gives the highest F1 on test split
"""
## In practical, model selection is conducted using validation data instead of testing data
