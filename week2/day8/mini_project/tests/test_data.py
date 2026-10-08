from src.data import generate_data, split_and_convert
import torch

def test_input_output():
    X, y = generate_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_convert(X, y)
    assert len(X) == len(y)
    assert X[1].size == 10
    assert torch.tensor(y).unsqueeze(1).shape == (len(y), 1)
    assert all(split.dtype==torch.float32 for split in [X_train, X_val, X_test, 
                                                     y_train, y_val, y_test])