import numpy as np

# Exercise 1: array inspection
X = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
"""
X.shape = (3, 4)
X.ndim = 2
X.size = 12
X[1, 2] = 7
X[:, 1] = [2, 6, 10]
X[1:, :2] = [[5, 6], [9, 10]]
"""
print(f"X.shape = {X.shape}, X.ndim = {X.ndim}, X.size = {X.size}")
print(f"X[1, 2] = {X[1, 2]}, X[:, 1] = {X[:, 1]}, X[1:, :2] = {X[1:, :2]}")

# Exercise 2: statistics
scores = np.array([
    [0.80, 0.90, 0.70],
    [0.60, 0.85, 0.95],
    [0.75, 0.80, 0.90],
    [0.90, 0.95, 0.85]
])
print(f"{scores.mean()}, mean for each column: {scores.mean(axis=0)}, \
      mean for each row: {scores.mean(axis=1)}")
print(f"{scores.max()}, {scores.min()}, {scores.std()}")

# Exercise 3: filtering
confidence = np.array([0.91, 0.43, 0.77, 0.95, 0.62, 0.88])
filtered = confidence[confidence>=0.8]
print(f"Confidence values >=0.8: {filtered}; {len(filtered)}")

# Exercise 4: normalisation
X = np.array([
    [10, 100],
    [20, 200],
    [30, 300],
    [40, 400]
])
X_standardised = (X - X.mean(axis=0)) / X.std(axis=0)
print(X_standardised)

# Exercise 5: linear model
X = np.array([
    [2.0, 3.0],
    [1.0, 5.0],
    [4.0, 2.0],
    [3.0, 1.0]
])
weights = np.array([0.7, -0.4])
bias = 0.2
prediction = X @ weights + bias
"""
X.shape = (4, 2)
weights.shape = (2, )
prediction.shape = (4, ) because X @ weights produces a shape of (4, )
"""
print(f"X.shape = {X.shape}, weights.shape = {weights.shape}, \
      prediction.shape = {prediction.shape}")
print(prediction)

# Exercise 6: classification
prediction = (prediction>=0).astype(np.int64)
print(prediction)
