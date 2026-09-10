import pandas as pd
from sklearn.model_selection import train_test_split

# Exercise 1: split reasoning
X = pd.DataFrame({
    "feature_1": range(100),
    "feature_2": range(100, 200)
})
y = pd.Series([0, 1] * 50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
"""
X.shape = (100, 2)
y.shape = (100, )
X_train.shape = (80, 2)
X_test.shape = (20, 2)
y_train.shape = (80, )
y_test.shape = (20, )
"""
print(f"{X.shape}, {y.shape}")
print(f"{X_train.shape}, {X_test.shape}")
print(f"{y_train.shape}, {y_test.shape}")

# Exercise 2: class distribution
print(y.value_counts(normalize=True))
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
X_train_unstratify, X_test_unstratify, y_train_unstratify, y_test_unstratify = train_test_split(X, y, test_size=0.2, random_state=42)
print("Unstratify y")
print(y.value_counts(normalize=True))
print(y_train_unstratify.value_counts(normalize=True))
print(y_test_unstratify.value_counts(normalize=True))
## Stratify makes train an test sets preserve approximately the same class distribution as the original.

# Exercise 3: scaling
X = pd.DataFrame({
    "age": [20, 25, 30, 35, 40, 45, 50, 55],
    "income": [
        20000, 35000, 42000, 50000,
        65000, 80000, 95000, 120000
    ]
})
from sklearn.preprocessing import StandardScaler
X_train, X_test = train_test_split(X, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_transform = scaler.transform(X_train)
X_test_transform = scaler.transform(X_test)
## X_test is not used in scaler fitting to aviod unseen data influence preprocess stage
print(f"{scaler.mean_}, {scaler.scale_}")

# Exercise 4: classifier
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
dataset = load_breast_cancer(as_frame=True)
print(dataset.data)
print(dataset.target)
print(dataset.target_names)
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
classifier = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")

# Exercise 5: probabilities
probabilities = classifier.predict_proba(X_test_scaled)
print(f"{probabilities.shape}, {probabilities[:5]}")
## The two output probabilities of each test sample determine how likely it is belong to.
positive_class_probabilities = probabilities[:, 1]

# Exercise 6: custom threshold
custom_predictions_70 = (positive_class_probabilities>=0.7).astype(int)
custom_predictions_50 = (positive_class_probabilities>=0.5).astype(int)
custom_predictions_30 = (positive_class_probabilities>=0.3).astype(int)
precision_70, recall_70, f1_70 = precision_score(y_test, custom_predictions_70), recall_score(y_test, custom_predictions_70), f1_score(y_test, custom_predictions_70)
precision_50, recall_50, f1_50 = precision_score(y_test, custom_predictions_50), recall_score(y_test, custom_predictions_50), f1_score(y_test, custom_predictions_50)
precision_30, recall_30, f1_30 = precision_score(y_test, custom_predictions_30), recall_score(y_test, custom_predictions_30), f1_score(y_test, custom_predictions_30)
print(f"postive-class threshold=0.7: {precision_70}, {recall_70}, {f1_70}")
print(f"postive-class threshold=0.5: {precision_50}, {recall_50}, {f1_50}")
print(f"postive-class threshold=0.3: {precision_30}, {recall_30}, {f1_30}")
"""
postive-class threshold=0.7: 0.9882352941176471, 0.9333333333333333, 0.96
postive-class threshold=0.5: 0.9888888888888889, 0.9888888888888889, 0.9888888888888889
postive-class threshold=0.3: 0.967741935483871, 1.0, 0.9836065573770492
"""

# Exercise 7: confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
"""
[[52  1]
 [ 1 89]]
TN: classify negative, while true class is indeed negative
FP: classify as positive, but in fact it is negative
FN: classify as negative, but in fact it is positive
TP: classify positive, while true class is indeed positive
"""