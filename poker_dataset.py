from torch.utils.data import Dataset

import kaggle_dataset_poker as kdp
import torch
import numpy as np
import os

class PreFlopDataset(Dataset):
    def __init__(self, poker_hands: list[kdp.PokerHand]) -> None:
        super().__init__()
        self.poker_hand_tensors = []
        for hand in poker_hands:
            try:
                hand_tensor, my_action = hand_to_tensor(hand)
            except Exception as e:
                continue
            self.poker_hand_tensors.append((hand_tensor, my_action))
        
    def __len__(self):
        return len(self.poker_hand_tensors)
    
    def __getitem__(self, index):
        if index < 0 or index >= self.__len__():
            raise IndexError("Index out of bounds")
        return self.poker_hand_tensors[index]

def get_cards_tensor(cards:list[kdp.Card]):
    """
    Convert a list of cards to a vector representation
    """
    tensor = np.concatenate([card.vector for card in cards])
    tensor = tensor.astype(np.float32)
    tensor = torch.from_numpy(tensor)
    return tensor

def action_to_tensor(action: kdp.PlayerAction):
    """
    Convert a poker action to a vector representation
    """
    tensor = torch.zeros(5, dtype=torch.float32)
    if action.player_action == 'folds':
        tensor[0] = 1
    elif action.player_action == 'calls':
        tensor[1] = 1
        tensor[3] = action.player_bet/1000
    elif action.player_action == 'raises' or action.player_action == 'allin':
        tensor[2] = 1
        tensor[3] = action.player_bet/1000
    tensor[4] = action.player_stack/1000
    return tensor

def get_prior_actions_tensor(prior_actions: list[kdp.PlayerAction]):
    """
    Convert a list of prior actions to a tensor representation
    """
    tensor = torch.zeros(10*5, dtype=torch.float32)
    for i, action in enumerate(prior_actions):
        index = action.order
        tensor[index:index+5] = action_to_tensor(action)
    return tensor

def hand_to_tensor(hand: kdp.PokerHand):
    """
    Convert a poker hand to a vector representation
    """
    
    # Get player position tensor
    player_position = torch.tensor([hand.player_position/10], dtype=torch.float32)
    
    # Get player stack to pot ratio tensor
    # player_stack_to_pot_ratio = torch.tensor([hand.], dtype=torch.float32)
    
    # Get player stack tensor
    player_stack = torch.tensor([hand.start_money/1000], dtype=torch.float32)
    
    # Get hole cards tensor
    hole_cards = get_cards_tensor(hand.hole_cards)
    
    # # Get community cards tensor
    # community_cards = get_cards_tensor(hand.community_cards)
    
    # Get prior actions tensor
    prior_actions = get_prior_actions_tensor(hand.prior_actions)
    
    # Concatenate all tensors
    hand_input_tensor = torch.cat([player_position, player_stack, hole_cards, prior_actions])
    
    # Get player action tensor
    my_action = action_to_tensor(hand.my_action)
    
    return hand_input_tensor, my_action

def read_and_parse_games_from_folder(folder_path):
    poker_hands = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                poker_hands += kdp.read_and_parse_games(file_path)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    return poker_hands

def check_hand_tensor_size(poker_hands):
    sizes = []
    ylabels = []
    for hand in poker_hands:
        try:
            tensor, ylabel = hand_to_tensor(hand)
        except:
            continue
        sizes.append(tensor.size())
        ylabels.append(ylabel.size())
    sizes = set(sizes)
    ylabels = set(ylabels)
    print("Input sizes: ", sizes)
    print("Output sizes: ", ylabels)

if __name__ == '__main__':
    filename = r'poker_data'
    poker_hands = read_and_parse_games_from_folder(filename)
    poker_hand_dataset = PreFlopDataset(poker_hands)
    #check_hand_tensor_size(poker_hands)
    print(len(poker_hand_dataset))
    hand = poker_hands[5]
    # print(hand.prior_actions)
    
    # max_start_money = max([hand.start_money for hand in poker_hands])
    # print(max_start_money)
