from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def build_logistic_regression():
    return Pipeline([('preprocessing', build_preprocessor()), 
                     ('LR', LogisticRegression(random_state=42, max_iter=1000))])

def build_decision_tree():
    return Pipeline([('preprocessing', build_preprocessor()), 
                     ('dt', DecisionTreeClassifier(max_depth=3, random_state=42))])

def build_random_forest():
    return Pipeline([('preprocessing', build_preprocessor()), 
                     ('rf', RandomForestClassifier(n_estimators=100, random_state=42))])

def get_candidate_models() -> dict:
    return {"logistic_regression": build_logistic_regression(),
            "decision_tree": build_decision_tree(),
            "random_forest": build_random_forest()}