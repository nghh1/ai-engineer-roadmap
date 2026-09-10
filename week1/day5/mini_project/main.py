from src.loading import load_data
from src.preprocessing import split_features_target
from src.training import build_model
from src.evaluation import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    df = load_data("./data/customer.csv")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    print(metrics)

if __name__ == "__main__":
    main()