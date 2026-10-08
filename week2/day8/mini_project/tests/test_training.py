from src.data import generate_data, split_and_convert, build_data_loader
from src.model import BinaryClassifier
from src.training import train_once, validate_once
import torch
from torch import nn

def test():
    torch.manual_seed(42)
    X, y = generate_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_convert(X, y)
    train_loader, validate_loader, test_loader = build_data_loader(X_train, X_val, X_test, y_train, y_val, y_test)
    model = BinaryClassifier(10)
    loss_fn  =nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial_loss, initial_validate_loss = 0, 0
    for i in range(30):
        if i == 0: 
            initial_loss = train_once(model, train_loader, loss_fn, optimiser, device='cpu')
            initial_validate_loss = validate_once(model, validate_loader, loss_fn, device='cpu')
        else:
            loss = train_once(model, train_loader, loss_fn, optimiser, device='cpu')
            loss_validate = validate_once(model, validate_loader, loss_fn, device='cpu')
    assert (loss < initial_loss) and (loss_validate < initial_validate_loss)