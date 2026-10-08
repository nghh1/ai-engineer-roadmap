import torch
from torch import nn
# Exercise 1: Tensor basics
X = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])
print(X.shape) # shape = (3, 3)
print(X.dtype) # torch.float32
print(X.ndim) # dimension = 2
print(f"mean = {X.mean()}, sum = {X.sum()}")
print(f"feature mean across data = {X.mean(dim=0)}, \
      mean across features = {X.mean(dim=1)}")

# Exercise 2: Matrix multiplication
X = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])
w = torch.tensor([0.5, -0.25])
product = X @ w # shape = (3, )
print(product) # product = [0.0, 0.5, 1.0]

# Exercise 3: autograd
x = torch.tensor(4.0, requires_grad=True)
y = 3 * x**2 + 2 * x + 1 
y.backward() # dy/dx=6 * x + 2; when x=4, dy/dx=26
print(x.grad)

# Exercise 4: linear layer shapes
layer = nn.Linear(4, 3)
X = torch.randn(10, 4)
output = layer(X) # output shape = (10, 3)
print(output.shape)
print(layer.weight.shape) # shape = (3, 4) where each output neuron connects with each input feature/neuron
print(layer.bias.shape) # shape = (3, ), each output neuron is associated with a bias

# Exercise 5: build a network
input_size, output_size = 4, 1
hidden_layer1, hidden_layer2 = 16, 8
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_layer1),
            nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU(),
            nn.Linear(hidden_layer2, output_size)
        )

    def forward(self, x):
        return self.model(x)
x = torch.randn(32, 4)
model = Network()
output = model(x)
print(output.shape) 
# shape = (32, 1) as expected, from shape (32, 4) -> (32, 16) -> (32, 8) -> (32, 1)

# Exercise 6: one manual training step
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])
y = torch.tensor([[0.0], [1.0], [1.0], [1.0]])
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    def forward(self, x):
        return self.model(x)
classifier = SimpleClassifier()
optimiser = torch.optim.Adam(classifier.parameters())
bce = nn.BCEWithLogitsLoss()
optimiser.zero_grad()
logits = classifier(X)
loss = bce(logits, y)
loss.backward()
optimiser.step()
logits_after = classifier(X)
loss_after = bce(logits_after, y)
print(f"loss before: {loss.item()}")
print(f"loss after: {loss_after.item()}")

# Exercise 7: train the OR classifier
classifier = SimpleClassifier()
optimiser = torch.optim.Adam(classifier.parameters())
bce = nn.BCEWithLogitsLoss()
for i in range(800):
    optimiser.zero_grad()
    logits = classifier(X)
    loss = bce(logits, y)
    loss.backward()
    optimiser.step()
classifier.eval()
with torch.no_grad():
    logits = classifier(X)
    prob = torch.sigmoid(logits)
    prediction = (prob >= 0.5).int()
print(f"prob: {prob}, prediction: {prediction}, actural: {y}")