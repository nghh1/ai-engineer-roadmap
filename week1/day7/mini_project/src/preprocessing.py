import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["customer_id", "churned"])
    y = df["churned"]
    return X, y

def build_preprocessor():
    numeric_features = ["age", "monthly_spend", "account_age_months", "support_tickets"]
    categorical_features = ["plan", "country"]
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("scaler", scaler, numeric_features), 
                              ("encoder", encoder, categorical_features)])
