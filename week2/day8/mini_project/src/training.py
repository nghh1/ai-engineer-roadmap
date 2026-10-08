import torch
from torch import nn

def train_once(model, dataloader, loss_fn, optimiser, device='cpu'):
    model.train()
    accum_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimiser.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimiser.step()
        accum_loss += loss.item()
    return accum_loss / len(dataloader)

def validate_once(model, dataloader, loss_fn, device='cpu'):
    model.eval()
    accum_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            accum_loss += loss.item()
        return accum_loss / len(dataloader)

def train_validate_model(model, train_loader, validate_loader, epochs, 
                         learning_rate, device):
    loss_fn =nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        loss = train_once(model, train_loader, loss_fn, optimiser, device)
        loss_validate = validate_once(model, validate_loader, loss_fn, device)
        if epoch % 10 == 0: print(f"Epoch {epoch}: loss = {loss}, validate_loss = {loss_validate}")