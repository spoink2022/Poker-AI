import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as v2

import math
from poker_dataset import PreFlopDataset, read_and_parse_games_from_folder
from poker_ai_model import CustomFCNetwork, train_model, evaluate_model, test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available: ", torch.cuda.is_available())

def filter_out_betsize(x, y):
    y = y[:, :3]
    return x, y

if __name__ == '__main__':
    filename = r'poker_data'
    poker_hands = read_and_parse_games_from_folder(filename)
    poker_hand_dataset = PreFlopDataset(poker_hands)
    
    test_fraction = 1/5
    test_len, train_len = math.floor(len(poker_hand_dataset)*test_fraction), math.ceil(len(poker_hand_dataset)*(1-test_fraction))
    test_dataset, train_dataset = torch.utils.data.random_split(poker_hand_dataset, [test_len, train_len])
    
    poker_ai_model = CustomFCNetwork(network_shape=[87, 64, 32, 3]).to(device=device)
    
    num_epochs = 60
    batch_size = 256
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(poker_ai_model.parameters(), lr=0.001)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_model(poker_ai_model, train_loader, optimizer, criterion, epochs=num_epochs)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print(test_model(poker_ai_model, test_loader, process_batch=filter_out_betsize))
    