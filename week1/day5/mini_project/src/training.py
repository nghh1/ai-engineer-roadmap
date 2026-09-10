from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.preprocessing import build_preprocessor

def build_model() -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression())
    ])
 