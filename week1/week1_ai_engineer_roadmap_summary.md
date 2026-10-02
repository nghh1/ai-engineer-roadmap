# AI Engineer Roadmap — Week 1 Summary

## Overview

Week 1 focused on building a strong practical foundation for AI engineering by progressing from core Python into data processing, classical machine learning, model evaluation, and hyperparameter tuning.

The progression across the week was:

```text
Day 1  → Python fundamentals
Day 2  → Intermediate Python + testing + modular design
Day 3  → NumPy + numerical thinking
Day 4  → Pandas + data preparation
Day 5  → Scikit-learn + first classification pipeline
Day 6  → Cross-validation + model selection
Day 7  → Hyperparameter tuning + search
```

By the end of Week 1, the workflow evolved from writing simple Python functions into building reusable machine-learning pipelines with preprocessing, multiple candidate models, cross-validation, hyperparameter tuning, automated model selection, tests, and final holdout evaluation.

---

# Day 1 — Python Fundamentals and a Simple Inference Pipeline

## Main Goal

Build confidence with core Python and use those fundamentals to construct a small inference-style pipeline.

## Topics Covered

- Variables and basic syntax
- Conditions and control flow
- Loops
- Functions
- Lists, dictionaries, tuples, and sets
- Classes
- Exceptions
- File handling
- Basic program structure

## Practical Exercises

The exercises focused on writing small pieces of logic using:

- `if` / `elif` / `else`
- loops
- functions with arguments and return values
- list and dictionary manipulation
- validation logic
- simple class usage
- exception handling

## Mini-Project — Simple Inference Pipeline

The first mini-project simulated a small prediction system.

The core flow became:

```text
records
   ↓
validate_record()
   ↓
preprocess_text()
   ↓
predict()
   ↓
collect predictions
   ↓
save results
```

The final design contained functions such as:

```python
def validate_record(record):
    ...
```

```python
def preprocess_text(text):
    return text.strip().lower()
```

```python
def predict(text):
    ...
```

```python
def run_pipeline(records):
    ...
```

The pipeline:

- checked whether each input record was valid
- skipped invalid records
- normalized text
- generated a fake sentiment prediction
- returned structured prediction records
- saved predictions to JSON

## Important Lessons

### Functions should have one clear responsibility

Instead of putting all logic into one large function, responsibilities were separated:

```text
validation
preprocessing
prediction
orchestration
```

This pattern later became very important when building ML projects.

### Data validation should happen before processing

Bad inputs should be handled before calling preprocessing or prediction logic.

### Small pipelines already resemble production AI systems

Even though the prediction function was fake, the structure resembles a real inference service:

```text
input
→ validate
→ preprocess
→ inference
→ output
```

## Day 1 Outcome

By the end of Day 1, a working Python inference pipeline was built from scratch.

---

# Day 2 — Intermediate Python, Modularity, Typing, Generators, and Testing

## Main Goal

Turn the Day 1 script into a better structured Python project and introduce engineering practices used in production systems.

## Topics Covered

- Iterators
- Generators
- Type hints
- `TypedDict`
- Modules and packages
- Relative imports
- Project structure
- `pytest`
- Exceptions
- Debugging
- Batch processing

## Refactoring the Day 1 Pipeline

The pipeline was reorganized into modules:

```text
pipeline/
├── models.py
├── batching.py
├── validation.py
├── preprocessing.py
├── prediction.py
└── pipeline.py
```

A separate `main.py` became the entry point.

## Typed Records

Structured record types were introduced:

```python
from typing import TypedDict

class InputRecord(TypedDict):
    id: int
    text: str

class PredictionRecord(TypedDict):
    id: int
    text: str
    prediction: str
```

This improved readability and made expected data structures explicit.

## Generator-Based Batching

A reusable batching function was created:

```python
def batch(items: list, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
```

This introduced the idea of yielding data incrementally instead of returning everything at once.

## Modular Pipeline

The production-style pipeline became conceptually:

```text
input records
      ↓
validation
      ↓
batch generator
      ↓
preprocessing
      ↓
prediction
      ↓
typed result records
```

## Testing

`pytest` was introduced and tests were written for:

- validation
- preprocessing
- prediction
- batching
- pipeline behavior

The final test run reached:

```text
19 passed
```

## Important Lessons

### Modular code is easier to test

Once logic was split across modules, each function could be tested independently.

### Type hints communicate intent

Type hints do not replace tests, but they make interfaces clearer.

### Generators are useful for batch-oriented AI workloads

The batching exercise introduced a pattern that later applies to:

- inference batches
- dataset streaming
- API processing
- large data pipelines

### Tests are part of engineering, not an optional extra

The project was no longer considered finished simply because it ran.

## Day 2 Outcome

A simple Python script was transformed into a modular, typed, tested package.

---

# Day 3 — NumPy and Numerical Thinking

## Main Goal

Develop the numerical intuition needed for machine learning.

## Topics Covered

- NumPy arrays
- Shapes
- `dtype`
- Indexing and slicing
- Aggregations
- Axes
- Reshaping
- Vectorization
- Broadcasting
- Matrix multiplication
- Basic classifier evaluation

## Array Shapes

An important distinction was learned:

```python
np.array([0.7, -0.4]).shape
```

returns:

```text
(2,)
```

not:

```text
(2, 1)
```

A one-dimensional NumPy array has no explicit row or column orientation.

## Matrix Multiplication

For:

```text
X.shape       = (4, 2)
weights.shape = (2,)
```

then:

```python
X @ weights
```

produces:

```text
(4,)
```

This is the mathematical pattern later used by linear models.

## Vectorization

Instead of manually looping through arrays, NumPy operations were used directly.

For example:

```python
(probabilities >= threshold)
```

produces predictions for an entire array at once.

## Mini-Project — Binary Classifier Evaluation

A small classifier-evaluation system was implemented manually.

Functions included:

```python
def predict_labels(probabilities, threshold):
    ...
```

```python
def confusion_counts(y_true, y_pred):
    ...
```

```python
def accuracy(y_true, y_pred):
    ...
```

```python
def precision(y_true, y_pred):
    ...
```

```python
def recall(y_true, y_pred):
    ...
```

```python
def f1score(precision, recall):
    ...
```

The confusion matrix components were calculated using NumPy boolean expressions:

```text
TP
TN
FP
FN
```

## Important Lessons

### Metrics are formulas, not magic functions

Before using sklearn metrics, the formulas were implemented manually.

### Thresholds affect classification behavior

A probability does not automatically become a class label.

A threshold converts:

```text
probability
     ↓
decision threshold
     ↓
0 or 1
```

This concept became important again on Days 5 and 6.

### Shape awareness matters

Many ML bugs come from misunderstanding dimensions.

## Day 3 Outcome

The numerical foundation for machine learning was established with NumPy, linear algebra, vectorization, and manually implemented classification metrics.

---

# Day 4 — Pandas and Data Preparation for Machine Learning

## Main Goal

Learn how to clean, transform, summarize, and prepare tabular data for ML.

## Topics Covered

- `Series`
- `DataFrame`
- `.loc`
- filtering
- sorting
- missing values
- duplicates
- type conversion
- `groupby`
- aggregation
- feature engineering
- CSV loading and saving
- testing data pipelines

## Data Cleaning Exercises

The exercises included:

- selecting rows and columns
- boolean filtering with `&` and `|`
- sorting
- grouped aggregation
- cleaning strings
- numeric conversion
- missing-value handling
- duplicate removal

Useful patterns included:

```python
.str.lower().str.strip()
```

```python
pd.to_numeric(..., errors="coerce")
```

```python
.fillna(...)
```

```python
.drop_duplicates()
```

## Feature Engineering

New features were created from existing data.

Examples included:

```text
latency_seconds
high_accuracy
requests_per_ms
```

This introduced the idea that ML performance depends not only on the model, but also on how useful inputs are represented.

## Mini-Project — ML Dataset Preparation Pipeline

The project structure was:

```text
day04/
├── data/
│   ├── raw.csv
│   └── processed.csv
├── src/
│   ├── loading.py
│   ├── cleaning.py
│   ├── features.py
│   └── summary.py
├── tests/
│   ├── test_cleaning.py
│   └── test_features.py
└── main.py
```

## Loading

```python
def load_data(path):
    return pd.read_csv(path)
```

## Cleaning

The cleaning stage handled:

- inconsistent model names
- string whitespace
- numeric conversion
- missing numeric values
- duplicates

## Feature Engineering

Features included:

```text
total_tokens
latency_seconds
tokens_per_second
```

Special handling was added for zero-latency rows to avoid invalid divisions.

## Grouped Summary

The project produced model-level summaries such as:

```text
request count
success rate
average latency
average token count
average throughput
```

## Testing

Tests verified:

- original DataFrames were not mutated
- expected features were created
- zero latency produced safe output
- calculations matched expected values

Final result:

```text
8 passed
```

## Important Lessons

### Data preparation deserves its own modules

Cleaning and feature engineering should not be hidden inside model-training code.

### Avoid mutating caller input when possible

Using:

```python
df.copy()
```

made functions safer.

### Edge cases should be explicitly tested

For example:

```text
latency = 0
```

needed special handling.

## Day 4 Outcome

A reusable Pandas data-preparation pipeline was built and tested.

---

# Day 5 — Scikit-Learn, Classification Pipelines, and Holdout Evaluation

## Main Goal

Train the first real machine-learning model using scikit-learn.

## Topics Covered

- features vs target
- train/test splitting
- stratification
- standardization
- one-hot encoding
- Logistic Regression
- `Pipeline`
- `ColumnTransformer`
- probability prediction
- classification metrics
- threshold tuning
- data leakage

## Train/Test Split

A standard split was introduced:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Why `random_state`?

It makes the split reproducible.

### Why `stratify=y`?

It approximately preserves the class distribution across train and test sets.

## Data Leakage

A crucial rule was learned:

```text
split first
then fit preprocessing on training data
```

Incorrect:

```text
fit scaler on all data
→ split later
```

Correct:

```text
split
→ fit preprocessing on training data
→ transform test data
```

## StandardScaler

Numerical features were standardized.

The scaler was fitted only on training data.

## OneHotEncoder

Categorical features were encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This made the pipeline robust to categories that were not seen during fitting.

## ColumnTransformer

Different preprocessing was applied to different column groups:

```text
X
├── numeric → StandardScaler
└── categorical → OneHotEncoder
```

## Pipeline

The preprocessing and classifier were combined:

```text
Pipeline
├── preprocessor
└── LogisticRegression
```

Then:

```python
model.fit(X_train, y_train)
```

automatically fitted preprocessing and the classifier correctly.

## Day 5 Exercises

The exercises covered:

- train/test split shapes
- stratification
- scaling
- Logistic Regression
- `predict_proba`
- threshold changes
- confusion matrix interpretation

A threshold experiment demonstrated:

```text
lower threshold
→ more positive predictions
→ higher recall
→ potentially lower precision
```

## Mini-Project — Customer Churn Classifier

Dataset columns:

```text
customer_id
age
monthly_spend
account_age_months
support_tickets
plan
country
churned
```

The model used:

```text
numeric features
→ StandardScaler

categorical features
→ OneHotEncoder

classifier
→ LogisticRegression
```

The test set remained separate from model fitting.

Final tests passed successfully.

## Important Lessons

### Preprocessing belongs inside the Pipeline

This prevents leakage and guarantees identical transformations during training and prediction.

### `predict()` and `predict_proba()` answer different questions

```text
predict()
→ class

predict_proba()
→ probability
```

### Accuracy alone is not enough

The project used:

- accuracy
- precision
- recall
- F1

## Day 5 Outcome

The first complete sklearn classification pipeline was built from raw tabular data through preprocessing, training, prediction, and evaluation.

---

# Day 6 — Cross-Validation, Overfitting, and Model Selection

## Main Goal

Learn how to estimate generalization performance more reliably and compare different model families.

## Topics Covered

- underfitting
- overfitting
- train vs test performance
- cross-validation
- `StratifiedKFold`
- CV mean and standard deviation
- Logistic Regression
- Decision Tree
- Random Forest
- ROC-AUC
- model selection
- holdout evaluation

## Overfitting Experiment

An unrestricted Decision Tree produced:

```text
train F1 = 1.0
test F1  ≈ 0.52
```

This clearly demonstrated overfitting.

A restricted tree with:

```python
max_depth=3
```

reduced training performance but improved generalization.

The lesson:

> Better training performance does not necessarily mean a better model.

## Cross-Validation

Five-fold CV was introduced.

Conceptually:

```text
Fold 1: validation
Fold 2: validation
Fold 3: validation
Fold 4: validation
Fold 5: validation
```

Every sample gets a chance to be part of validation.

The main outputs were:

```text
mean CV score
standard deviation
```

Mean estimates average performance.

Standard deviation gives information about stability across folds.

## StratifiedKFold

For classification:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

was used to preserve class proportions across folds.

A useful indexing lesson was learned:

```python
y_train.iloc[train_idx]
```

must be used when fold indices refer to positional locations inside `y_train`.

## ROC-AUC

ROC-AUC was added as a probability-based metric:

```python
roc_auc_score(y_test, y_prob)
```

Unlike F1 at a single threshold, ROC-AUC evaluates ranking behavior across thresholds.

## Model Comparison

Three candidate models were compared:

```text
Logistic Regression
Decision Tree
Random Forest
```

using cross-validation on training data only.

The workflow became:

```text
training portion
     ↓
cross-validation
     ↓
compare models
     ↓
select best candidate
     ↓
fit on full training data
     ↓
evaluate once on test data
```

## Mini-Project — Model Selection Report

The project introduced reusable functions such as:

```python
def cross_validate_model(model, X, y, cv):
    ...
```

and:

```python
def compare_models(models, X, y, cv):
    ...
```

A DataFrame was produced containing:

```text
model
mean_f1
std_f1
```

The winner was selected dynamically rather than hardcoded.

Final model selection used CV only on:

```text
X_train
y_train
```

Then the selected model was evaluated on:

```text
X_test
y_test
```

## Important Lessons

### The test set is not for model selection

Using the test set repeatedly turns it into validation data.

### CV scores are estimates

A strong CV score does not guarantee identical holdout performance.

### Model complexity affects generalization

The unrestricted Decision Tree experiment made the bias/variance tradeoff visible.

## Day 6 Outcome

A proper model-selection workflow was built using stratified cross-validation and a final untouched holdout set.

---

# Day 7 — Hyperparameter Tuning and Automated Search

## Main Goal

Move beyond model-family selection and optimize the configuration of each candidate model.

## Topics Covered

- model parameters vs hyperparameters
- pipeline parameter naming
- `GridSearchCV`
- `RandomizedSearchCV`
- `cv_results_`
- model search spaces
- best estimator
- dynamic model selection
- tuned vs baseline comparison
- test-set discipline

## Parameters vs Hyperparameters

Parameters are learned during training.

Examples:

```text
Logistic Regression coefficients
tree split structure
```

Hyperparameters are selected before fitting.

Examples:

```text
C
max_depth
min_samples_split
n_estimators
```

## Pipeline Parameter Names

Pipeline parameters were accessed using:

```text
step_name__parameter_name
```

Examples:

```text
LR__C
dt__max_depth
rf__n_estimators
```

The prefix comes from the Pipeline step name.

## GridSearchCV

Grid search was used for smaller search spaces.

Decision Tree parameters included:

```python
{
    "dt__max_depth": [2, 3, 4, 5, None],
    "dt__min_samples_leaf": [1, 2, 4],
    "dt__min_samples_split": [2, 5, 10],
}
```

The best Decision Tree configuration improved CV F1 over the baseline.

## Logistic Regression Tuning

The search space was:

```python
{
    "LR__C": [0.01, 0.1, 1, 10, 100]
}
```

The best value was:

```text
C = 1
```

which was already the default.

This demonstrated an important lesson:

> Hyperparameter tuning does not always improve a model.

## RandomizedSearchCV

Random Forest had a much larger search space, so randomized search was used.

Parameters included:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

Rather than evaluating every possible combination, only a fixed number of random configurations were sampled.

This greatly reduced the number of model fits.

## Baseline vs Tuned Results

The experiments showed approximately:

```text
Logistic Regression
baseline ≈ 0.602
tuned    ≈ 0.602

Decision Tree
baseline ≈ 0.556
tuned    ≈ 0.588

Random Forest
baseline ≈ 0.637
tuned    ≈ 0.679
```

So:

```text
Logistic Regression
→ no improvement

Decision Tree
→ moderate improvement

Random Forest
→ strongest improvement
```

## Inspecting Search Results

`cv_results_` was converted into a DataFrame.

Important columns included:

```text
mean_test_score
std_test_score
rank_test_score
```

This made it possible to understand the search landscape instead of only trusting `best_params_`.

## Mini-Project — Hyperparameter Search Pipeline

The final Week 1 project extended the model-selection workflow into a reusable tuning system.

The architecture became:

```text
full dataset
    ↓
train/test split
    ↓
training data only
    ↓
┌─────────────────────────┐
│ tune Logistic Regression│
│ tune Decision Tree      │
│ tune Random Forest      │
└─────────────┬───────────┘
              ↓
     compare best CV F1
              ↓
     select winner
              ↓
      best_estimator_
              ↓
       untouched test
              ↓
     final evaluation
```

The winner was selected dynamically from the tuning result table.

The test set remained outside all hyperparameter searches.

## Important Lessons

### Hyperparameter tuning is another form of model selection

The validation/CV data decides:

```text
which model
+
which configuration
```

### Search must happen only on training data

The test set must remain untouched.

### `best_estimator_` is already fitted

Because sklearn search objects use `refit=True` by default, the winning configuration is refitted on the full training portion after CV.

### Automated search should still be interpretable

Search results should be inspected, not blindly accepted.

## Day 7 Outcome

A reusable hyperparameter-search system was built using GridSearchCV and RandomizedSearchCV, including automated winner selection and final holdout evaluation.

---

# Week 1 — Engineering Skills Developed

Across the seven days, the following software-engineering skills were repeatedly practiced.

## Modular Design

Projects were split into modules such as:

```text
loading.py
cleaning.py
preprocessing.py
models.py
training.py
validation.py
tuning.py
evaluation.py
```

Each module had a specific responsibility.

## Testing

`pytest` was used throughout the week.

Tests covered:

- pure functions
- data cleaning
- feature generation
- batch generators
- sklearn pipelines
- evaluation metrics
- validation functions
- model search objects

The mindset shifted from:

```text
"It runs"
```

to:

```text
"It runs and its important behavior is verified"
```

## Reproducibility

Repeated use of:

```python
random_state=42
```

made experiments reproducible.

## Separation of Responsibilities

The projects increasingly separated:

```text
loading
preprocessing
training
validation
evaluation
orchestration
```

This is a major production engineering habit.

## Avoiding Data Leakage

One of the most important principles learned during Week 1 was:

```text
training data
→ fit parameters

validation / CV
→ choose models and hyperparameters

test data
→ final evaluation only
```

---

# Week 1 — Machine Learning Concepts Learned

By the end of Week 1, the following ML concepts had been implemented in code rather than only studied theoretically.

## Data Preparation

- numeric cleaning
- categorical handling
- missing values
- feature engineering
- train/test separation
- standardization
- one-hot encoding

## Models

- Logistic Regression
- Decision Tree
- Random Forest

## Evaluation

- accuracy
- precision
- recall
- F1
- confusion matrix
- ROC-AUC

## Probability and Thresholds

The relationship between:

```text
predicted probability
        ↓
threshold
        ↓
class prediction
```

was explored directly.

## Generalization

The difference between:

```text
training performance
```

and:

```text
unseen-data performance
```

was demonstrated experimentally.

## Cross-Validation

Cross-validation was used to produce more robust estimates than a single validation split.

## Hyperparameter Tuning

Both exhaustive and randomized search strategies were implemented.

---

# Week 1 — End-to-End Workflow Achieved

At the beginning of the week, the workflow was:

```text
write a Python function
```

By the end of the week, it had become:

```text
raw tabular data
        ↓
load
        ↓
clean / validate
        ↓
feature / target separation
        ↓
train / test split
        ↓
Pipeline
├── preprocessing
│   ├── StandardScaler
│   └── OneHotEncoder
│
└── classifier
        ↓
cross-validation
        ↓
compare candidate models
        ↓
hyperparameter search
        ↓
select best model + configuration
        ↓
best_estimator_
        ↓
untouched holdout test
        ↓
accuracy / precision / recall / F1 / ROC-AUC
        ↓
tested, modular project
```

---

# Key Week 1 Principles

## 1. Separate training, validation, and testing

```text
training
→ learn parameters

validation / CV
→ choose models and hyperparameters

test
→ final unbiased evaluation
```

## 2. Put preprocessing inside sklearn Pipelines

This reduces leakage and keeps training and inference transformations consistent.

## 3. Never judge a model using training performance alone

A model can memorize the training set and still generalize poorly.

## 4. Prefer reusable functions over duplicated logic

Examples:

```python
cross_validate_model(model, ...)
```

instead of:

```text
cross_validate_logistic(...)
cross_validate_tree(...)
cross_validate_forest(...)
```

## 5. Hyperparameter tuning does not guarantee improvement

The Logistic Regression experiment showed that the default configuration can already be competitive.

## 6. Test sets should remain untouched

Repeatedly checking the test set while developing effectively turns it into validation data.

## 7. Metrics must match the problem

Accuracy alone may hide poor positive-class performance.

Precision, recall, F1, and ROC-AUC provide different perspectives.

---

# Week 1 Final Status

```text
Day 1  ✅ Python fundamentals
Day 2  ✅ Intermediate Python + testing
Day 3  ✅ NumPy + numerical thinking
Day 4  ✅ Pandas + ML data preparation
Day 5  ✅ Scikit-learn classification pipeline
Day 6  ✅ Cross-validation + model selection
Day 7  ✅ Hyperparameter tuning + automated search
```

Week 1 successfully established the Python, data-processing, software-engineering, and classical machine-learning foundations needed for the next phase of the AI Engineer roadmap.

---

# Transition to Week 2

Week 1 primarily answered:

> How do I build, evaluate, compare, and tune traditional machine-learning systems correctly?

The next stage of the roadmap can now build on these foundations without needing to re-learn:

- Python project organization
- data handling
- preprocessing pipelines
- evaluation discipline
- cross-validation
- hyperparameter search
- testing
- leakage prevention

These skills will continue to apply when moving into more advanced AI engineering topics in Week 2.
