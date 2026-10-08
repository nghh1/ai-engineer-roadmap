import torch
from src.model import BinaryClassifier

def test_model_output_shape():
    X = torch.randn(8, 10)
    model = BinaryClassifier(10)
    output = model(X)
    assert output.shape == (8, 1)