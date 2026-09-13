import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["customer_id", "churned"])
    y = df["churned"]
    return X, y

def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "age", "monthly_spend", 
        "account_age_months", "support_tickets"
    ]
    categorical_features = ["plan", "country"]
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("numeric", scaler, numeric_features), 
                              ("categorical", encoder, categorical_features)])