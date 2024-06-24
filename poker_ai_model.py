import kaggle_dataset_poker as kdp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomFCNetwork(nn.Module):
    def __init__(self, network_shape = [28*28, 90, 30, 1], act_func = F.relu):
        super(CustomFCNetwork, self).__init__()
        self.act_func = act_func
        
        if len(network_shape) < 2:
            raise ValueError("Network shape must have at least 2 layers")
        self.output_layer = nn.Linear(network_shape[-2], network_shape[-1])
        
        self.hidden_layers = nn.ModuleList()
        for i in range(0, len(network_shape)-2):
            self.hidden_layers.append(nn.Linear(network_shape[i], network_shape[i+1]))
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.act_func(layer(x))
        
        x = self.output_layer(x)
        return x
    
def mnist_process_batch(x, y):
    x = x.view(x.shape[0], -1)/255
    y = torch.nn.functional.one_hot(y, num_classes=10).float()
    return x, y
    
def train_model(model, train_loader, optimizer, criterion, epochs=10, process_batch=None):
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for i, (x, y) in enumerate(tqdm(train_loader, desc="Batches", leave=True)):
            optimizer.zero_grad()
            if process_batch:
                x, y = process_batch(x, y)
            x, y = x.to(device=device), y.to(device=device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            # if i % 100 == 0:
            #     print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
            
def evaluate_model(model, test_loader, criterion, process_batch=None):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for x, y in test_loader:
            if process_batch:
                x, y = process_batch(x, y)
            x, y = x.to(device=device), y.to(device=device)
            y_pred = model(x)
            total_loss += criterion(y_pred, y).item()
        return total_loss/len(test_loader)
    
def test_model(model, test_loader, process_batch=None):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for x, y in test_loader:
            if process_batch:
                x, y = process_batch(x, y)
            x, y = x.to(device=device), y.to(device=device)
            y_pred = model(x)
            _, predictions = y_pred.max(1)
            _, y = y.max(1)
            total_correct += (predictions == y).sum()
            total_samples += y.size(0)
        return total_correct/total_samples
    
def mnist_check_model():
    mnist_data = datasets.MNIST("data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_fraction = 4/5
    test_len, train_len = math.floor(len(mnist_data)*test_fraction), math.ceil(len(mnist_data)*(1-test_fraction))
    test_dataset, train_dataset = torch.utils.data.random_split(mnist_data, [test_len, train_len])
    
    model = CustomFCNetwork(network_shape=[28*28, 30, 10]).to(device=device)

    num_epochs = 5
    batch_size = 5

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_model(model, data_loader, optimizer, criterion, epochs=num_epochs, process_batch=mnist_process_batch)
    
    model = CustomFCNetwork()
    x = torch.randn(1, 28*28)
    print(model(x))
    
if __name__ == "__main__":
    mnist_check_model()





