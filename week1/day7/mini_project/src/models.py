from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import build_preprocessor
from sklearn.pipeline import Pipeline

def build_logistic_regression():
    return Pipeline([("Preprocessor", build_preprocessor()), ("LR", LogisticRegression())])

def build_decision_tree():
    return Pipeline([("Preprocessor", build_preprocessor()), ("DT", DecisionTreeClassifier())])

def build_random_forest():
    return Pipeline([("Preprocessor", build_preprocessor()), ("RF", RandomForestClassifier())])

def get_candidate_models() -> dict:
    return {
        "logistic_regression": build_logistic_regression(),
        "decision_tree": build_decision_tree(),
        "random_forest": build_random_forest()
    }
