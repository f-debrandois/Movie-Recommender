
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Training function
def train(model, criterion, optimizer, train_loader):
    model.to(device)
    model.train()
    
    train_losses = []
    train_acc = []

    for inputs, _, targets in tqdm(train_loader, desc="Training", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_losses.append(loss.item())
        
        # Compute accuracy
        _, preds = torch.max(outputs, 1)
        correct = (preds == targets).sum().item()
        acc = correct / targets.size(0)
        train_acc.append(acc)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return np.mean(train_losses), np.mean(train_acc)


# Test function
def test(model, criterion, test_loader):
    model.eval()
    
    test_losses = []
    test_acc = []
    
    with torch.no_grad():
        for inputs, _, targets in tqdm(test_loader, desc="Testing", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.append(loss.item())
            
            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            correct = (preds == targets).sum().item()
            acc = correct / targets.size(0)
            test_acc.append(acc)
            
    return np.mean(test_losses), np.mean(test_acc)



# Training loop
def training_loop(model, criterion, optimizer, num_epochs, train_loader, test_loader):

    train_losses, test_losses = [], []
    train_acc, test_acc = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")
        train_loss, train_accuracy = train(model, criterion, optimizer, train_loader)
        test_loss, test_accuracy = test(model, criterion, test_loader)
        print("-----------------------------------")
        print(f"Train loss: {train_loss:.4f} - Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy*100:.2f}%")
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)
        test_losses.append(test_loss)
        test_acc.append(test_accuracy)
        print("-----------------------------------")
        print()
   
    #torch.save(model.state_dict(), "model")

    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()