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

def train_with_hyperparameters(network_shape, num_epochs, batch_size, learning_rate, poker_hand_dataset):    
    test_fraction = 1/10
    test_len, train_len = math.floor(len(poker_hand_dataset)*test_fraction), math.ceil(len(poker_hand_dataset)*(1-test_fraction))
    test_dataset, train_dataset = torch.utils.data.random_split(poker_hand_dataset, [test_len, train_len])
    
    poker_ai_model = CustomFCNetwork(network_shape=network_shape).to(device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(poker_ai_model.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_model(poker_ai_model, train_loader, optimizer, criterion, epochs=num_epochs)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return test_model(poker_ai_model, test_loader, process_batch=filter_out_betsize)

import pandas as pd

if __name__ == '__main__':
    filename = r'poker_data'
    poker_hands = read_and_parse_games_from_folder(filename)
    poker_hand_dataset = PreFlopDataset(poker_hands)
    
    print(poker_hand_dataset[0])
    
    num_epochs = 100
    batch_size_values = [32, 64, 128, 256]
    learning_rate_values = [0.1, 0.01, 0.001, 0.0001]
    network_shape_values = [[81, 32, 16, 3], [81, 64, 32, 3], [81, 128, 64, 3], [81, 256, 128, 3]]
    
    max_test_acc = 0
    
    test_acc_dicts = {}
    
    for network_shape in network_shape_values:
        test_acc_dicts[f"{network_shape}"] = {}
        for learning_rate in learning_rate_values:
            if "Batch Size" not in test_acc_dicts[f"{network_shape}"]:
                test_acc_dicts[f"{network_shape}"]["Batch Size"] = batch_size_values[:]
            test_acc_dicts[f"{network_shape}"][learning_rate] = []
            for batch_size in batch_size_values:
                test_acc = train_with_hyperparameters(network_shape, num_epochs, batch_size, learning_rate, poker_hand_dataset)
                test_acc_dicts[f"{network_shape}"][learning_rate].append(test_acc)
                
                print("Batch size: {batch_size}, Learning rate: {learning_rate}, Network shape: {network_shape}, Test Accuracy{test_acc}".format(batch_size=batch_size, learning_rate=learning_rate, network_shape=network_shape, test_acc=test_acc))
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    print("New max test accuracy: ", max_test_acc)
    
    df1 = pd.DataFrame(test_acc_dicts[f"{network_shape_values[0]}"])
    df2 = pd.DataFrame(test_acc_dicts[f"{network_shape_values[1]}"])
    df3 = pd.DataFrame(test_acc_dicts[f"{network_shape_values[2]}"])
    df4 = pd.DataFrame(test_acc_dicts[f"{network_shape_values[3]}"])
    
    df1.to_csv("test_acc_results_nsv1.csv")
    df2.to_csv("test_acc_results_nsv2.csv")
    df3.to_csv("test_acc_results_nsv3.csv")
    df4.to_csv("test_acc_results_nsv4.csv")
    
    pd.concat([pd.concat([df1, df2], axis=1), pd.concat([df3, df4], axis=1)]).to_csv('test_results.csv')
    
    
    

    