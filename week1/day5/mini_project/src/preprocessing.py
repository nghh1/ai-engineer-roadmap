import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["churned", "customer_id"])
    y = df["churned"]
    return X, y

def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["age", "monthly_spend", "account_age_months", "support_tickets"]
    categorical_features = ["plan", "country"]
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", scaler, numeric_features), ("cat", encoder, categorical_features)])
