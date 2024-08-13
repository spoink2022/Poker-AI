import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

class CustomFCNetwork(nn.Module):
    # Input size formerly 223
    def __init__(self, input_size = 264, hidden_shape = [90, 30], output_shapes = [11, 1], act_func = F.relu):
        super(CustomFCNetwork, self).__init__()
        self.act_func = act_func
        
        if len(hidden_shape) < 1 or len(output_shapes) != 2:
            raise ValueError("Hidden shape must be length larger than 1, output shape must be length 2")
        
        self.input_layer = nn.Linear(input_size, hidden_shape[0])
        
        self.hidden_layers = nn.ModuleList()
        for i in range(0, len(hidden_shape)-1):
            self.hidden_layers.append(nn.Linear(hidden_shape[i], hidden_shape[i+1]))
            
        self.action_layer = nn.Linear(hidden_shape[-1], output_shapes[0])
        self.bet_size_layer = nn.Linear(hidden_shape[-1], output_shapes[1])
    
    def forward(self, x):
        x = self.act_func(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.act_func(layer(x))
        
        bet_size = self.bet_size_layer(x)
        action_output = self.action_layer(x)
        return action_output, bet_size
    
def get_accuracy(predicted, label_data):
    action_label, bet_size_label = label_data
    action_output, bet_size = predicted
    
    predicted_action = torch.argmax(F.softmax(action_output, dim=1), dim=1)
    action_label_one_hot = torch.argmax(action_label, dim=1)
    num_corr = (predicted_action == action_label_one_hot).sum().item()
    total_num = len(predicted)
        
    return num_corr, total_num
          
def train_per_epoch(model, optimizer, dataloader):
    model.train()
    total_loss, total_corr, total_num = 0, 0, 0
    for input_tensor, action_label, bet_size_label, baseline_inputs, game_idx in tqdm(dataloader, desc='Batches', leave=False):
        action_output, bet_size = model(input_tensor)
        loss_action = nn.CrossEntropyLoss()(action_output, action_label)
        loss_bet_size = nn.MSELoss()(bet_size, bet_size_label)
        loss = loss_action + loss_bet_size
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_corr, _ = get_accuracy((action_output, bet_size), (action_label, bet_size_label))
        total_corr += num_corr
        total_loss += loss.item()
        total_num += len(input_tensor)
        
    return total_loss/total_num, total_corr/total_num

def evaluate(model, dataloader):
    model.eval()
    total_loss, total_corr, total_num = 0, 0, 0
    for input_tensor, action_label, bet_size_label, baseline_inputs, game_idx in tqdm(dataloader, desc='Evaluate', leave=False):
        action_output, bet_size = model(input_tensor)
        loss_action = nn.CrossEntropyLoss()(action_output, action_label)
        loss_bet_size = nn.MSELoss()(bet_size, bet_size_label)
        loss = loss_action + loss_bet_size
        
        num_corr, _ = get_accuracy((action_output, bet_size), (action_label, bet_size_label))
        total_corr += num_corr
        total_loss += loss.item()
        total_num += len(input_tensor)
    
    return total_loss/total_num, total_corr/total_num

def train(model, optimizer, train_dataloader, val_dataloader, num_epochs, display_progress = False):
    
    losses = {'Train': [], 'Validation': []}
    accs = {'Train': [], 'Validation': []}
    
    for epoch in tqdm(range(num_epochs), desc='Epochs', leave=display_progress):
        train_loss_avg, train_acc = train_per_epoch(model, optimizer, train_dataloader)
        val_loss_avg, val_acc = evaluate(model, val_dataloader)
        
        losses['Train'].append(train_loss_avg)
        losses['Validation'].append(val_loss_avg)
        accs['Train'].append(train_acc)
        accs['Validation'].append(val_acc)
        
        if display_progress:
            print("-----------------------------------------------------------------")
            print(f"Train loss:\t {train_loss_avg}, \t Train accuracy:\t {train_acc}")
            print(f"Valid loss:\t {val_loss_avg}, \t Valid accuracy:\t {val_acc}")
        
    return losses, accs

def train_with_hyperparameters(train_dataset, val_dataset, hidden_shape, learning_rate, batch_size, num_epochs=100, display_progress=False):
    
    model = CustomFCNetwork(hidden_shape = hidden_shape)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
    
    losses, accs = train(model, optimizer, train_dataloader, val_dataloader, num_epochs, display_progress)
    return losses, accs, model

import itertools
def tuning_hyperparameters(train_dataset, val_dataset, batch_sizes=None, learning_rates=None, hidden_shapes=None, num_epochs=20):
    # Test all permutation of available hyperparameters

    # Get list of available list of hyperparameters to test
    all_hps = [batch_sizes, learning_rates, hidden_shapes, num_epochs]
    tuning_hps = [hps for hps in all_hps if isinstance(hps, list)]
    is_hps_list = [isinstance(hps, list) for hps in all_hps]

    # Store training curve graphs
    acc_graphs = []
    labels = []

    # Test model for each combination of hyperparameters
    for hps_comb in tqdm(itertools.product(*tuning_hps), desc='Hyperparameters', leave=True):
        index = 0

        batch_size, learning_rate, hidden_shape, num_epochs = 64, 0.001, [128, 64], 50

        if is_hps_list[0]:
            batch_size = hps_comb[index]
            index += 1
        if is_hps_list[1]:
            learning_rate = hps_comb[index]
            index += 1
        if is_hps_list[2]:
            hidden_shape = hps_comb[index]
            index += 1
        if is_hps_list[3]:
            num_epochs = hps_comb[index]
            index += 1

        # Store training curve data
        losses, accs, model = train_with_hyperparameters(train_dataset, val_dataset, hidden_shape, learning_rate, batch_size, num_epochs)
        acc_graphs.append((accs['Train'], accs['Validation']))
        labels.append(f"Batch Size: {batch_size}, Learning Rate: {learning_rate},\n Hidden Size: {hidden_shape}")

    return acc_graphs, labels

import math
import textwrap

def plot_train_valid_list(train_valid_list, labels=None):
    # Calculate rows and columns
    num_tested = len(train_valid_list)
    cols = 3
    rows = math.ceil(num_tested / cols)

    # Set max title size and graph size
    fig, axs = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    max_title_length = 40

    # Plot graph for each tested model
    for i, (train_list, valid_list) in enumerate(train_valid_list):
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        ax.plot(train_list, marker='o', linestyle='-', label='Train')
        ax.plot(valid_list, marker='x', linestyle='--', label='Valid')
        # Set titles
        if labels and i < len(labels):
            wrapped_title = "\n".join(textwrap.wrap(labels[i], max_title_length))
            ax.set_title(wrapped_title)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()

    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        row = j // cols
        col = j % cols
        fig.delaxes(axs[row, col] if rows > 1 else axs[col])

    plt.tight_layout(pad=2.0)  # Adjust padding to make layout more compact
    plt.show()

import pandas as pd
from IPython.display import display
def display_train_valid_list(train_valid_list, labels=None, avg_over=10):
    # Displays the average of the last 10 values for each valid and train values.
    df_dict = {"Labels": [], "Train": [], "Valid": []}

    # Calculate the average of the last 10 values for each list
    for i, (train_list, valid_list) in enumerate(train_valid_list):
        df_dict["Labels"].append(labels[i])
        df_dict["Train"].append(sum(train_list[-avg_over:]) / avg_over)
        df_dict["Valid"].append(sum(valid_list[-avg_over:]) / avg_over)

    # Set DataFrame
    df = pd.DataFrame(df_dict)
    df = df.sort_values(by='Valid', ascending=False).reset_index(drop=True)

    # Set display options
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_rows', None)

    # Display the DataFrame
    display(df)

    # Reset display options to default
    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.expand_frame_repr')
    pd.reset_option('display.max_rows')

import matplotlib.pyplot as plt
def plot_curve(train_data, valid_data, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_data, label='train')
    plt.plot(valid_data, label='valid')
    plt.title(f'Training Curve: {title}')
    plt.legend()
    plt.show()

        
        