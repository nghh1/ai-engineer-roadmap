from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_validate
import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_data():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, 
                            n_redundant=2, n_classes=2, random_state=42)
    return X, y

def split_and_convert(X, y):
    X_trainVal, X_test, y_trainVal, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size=0.2, 
                                                      random_state=42, stratify=y_trainVal)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_data_loader(X_train: torch.Tensor, X_val: torch.Tensor, X_test: torch.Tensor, 
                  y_train: torch.Tensor, y_val: torch.Tensor, y_test: torch.Tensor):
    train_dataset = TensorDataset(X_train, y_train)
    validation_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, validation_loader, test_loader
