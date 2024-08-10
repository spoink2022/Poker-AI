import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as v2

import poker_model as pm
from poker_dataset import PokerDataset

filepath = r'C:\Users\kctak\Documents\Code\PokerAI\APS360 Progress\poker_data'
dataset = PokerDataset(filepath, "IlxxxlI")

# Define fraction of the entire dataset for training and validation
train_frac, val_frac = 0.7, 0.15
test_frac = 1 - train_frac - val_frac # The rest is for testing.

# Randomly split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_frac, val_frac, test_frac])
print(f"Train Dataset Length: {len(train_dataset)}, \nValidation Dataset Length: {len(val_dataset)}, \nTest Dataset Length: {len(test_dataset)}")

hidden_shape = [64, 32]
learning_rate = 0.001
batch_size = 512
num_epochs = 30

losses, accs = pm.train_with_hyperparameters(train_dataset, val_dataset, hidden_shape, learning_rate, batch_size, \
                                            num_epochs=num_epochs, display_progress=True)

hidden_shapes = [[64, 32], [128, 64], [256, 128]]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [32, 64, 128, 256, 512]

acc_graphs, labels = pm.tuning_hyperparameters(batch_sizes, learning_rates, hidden_shapes, num_epochs=50)
pm.plot_train_valid_list(acc_graphs, labels)

pm.plot_curve(losses['Train'], losses['Validation'], "Losses")
pm.plot_curve(accs['Train'], accs['Validation'], "Accuracy")