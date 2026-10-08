import torch
from src.data import generate_data, split_and_convert, build_data_loader
from src.model import BinaryClassifier
from src.training import train_validate_model
from src.evaluation import predict, evaluate_predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 
                        ('mps' if torch.mps.is_available() else 'cpu'))
    X, y = generate_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_convert(X, y)
    train_loader, valiate_loader, test_loader = build_data_loader(X_train, X_val, X_test, y_train, y_val, y_test)
    model = BinaryClassifier(input_size=10).to(device)
    train_validate_model(model, train_loader, valiate_loader, epochs=400, 
                         learning_rate=1e-4, device=device)
    probs, predictions = predict(model, X_test, device=device)
    results = evaluate_predictions(y_test, predictions.cpu())
    print(results)

if __name__ == '__main__':
    main()