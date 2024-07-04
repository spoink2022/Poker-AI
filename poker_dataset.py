from torch.utils.data import Dataset

import kaggle_dataset_poker as kdp
import torch
import numpy as np
import os

class PreFlopDataset(Dataset):
    def __init__(self, poker_hands: list[kdp.PokerHand]) -> None:
        super().__init__()
        self.poker_hand_tensors = []
        self.myaction_tensors = []
        for hand in poker_hands:
            try:
                hand_tensor, my_action = hand_to_tensor(hand)
            except Exception as e:
                continue
            self.poker_hand_tensors.append(hand_tensor)
            self.myaction_tensors.append(my_action)
        
        self.poker_hand_tensors = torch.stack(self.poker_hand_tensors)
        self.myaction_tensors = torch.stack(self.myaction_tensors)    
        
        normalize_dataset(self.poker_hand_tensors)
        normalize_dataset(self.myaction_tensors)
        
        self.poker_hand_tensors = torch.where(torch.isnan(self.poker_hand_tensors), torch.zeros_like(self.poker_hand_tensors), self.poker_hand_tensors)
        self.myaction_tensors = torch.where(torch.isnan(self.myaction_tensors), torch.zeros_like(self.myaction_tensors), self.myaction_tensors)
        
    def __len__(self):
        return len(self.poker_hand_tensors)
    
    def __getitem__(self, index):
        if index < 0 or index >= self.__len__():
            raise IndexError("Index out of bounds")
        return self.poker_hand_tensors[index], self.myaction_tensors[index]

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
        tensor[3] = action.player_bet
    elif action.player_action == 'raises' or action.player_action == 'allin':
        tensor[2] = 1
        tensor[3] = action.player_bet
    tensor[4] = action.player_stack
    return tensor

def get_prior_actions_tensor(prior_actions: list[kdp.PlayerAction]):
    """
    Convert a list of prior actions to a tensor representation
    """
    num_players = 9
    tensor = torch.zeros(num_players*5, dtype=torch.float32)
    for i, action in enumerate(prior_actions):
        index = (action.order-1)*5
        tensor[index:index+5] = action_to_tensor(action)
    return tensor

def hand_to_tensor(hand: kdp.PokerHand):
    """
    Convert a poker hand to a vector representation
    """
    
    # Get player position tensor
    player_position = torch.tensor([hand.player_position], dtype=torch.float32)
    
    # Get player stack to pot ratio tensor
    # player_stack_to_pot_ratio = torch.tensor([hand.], dtype=torch.float32)
    
    # Get player stack tensor
    player_stack = torch.tensor([hand.start_money], dtype=torch.float32)
    
    # Get hole cards tensor
    hole_cards = get_cards_tensor(hand.hole_cards)
    
    # # Get community cards tensor
    # community_cards = get_cards_tensor(hand.community_cards)
    
    # Get prior actions tensor
    prior_actions = get_prior_actions_tensor(hand.prior_actions)
    
    # Get player action tensor
    my_action = action_to_tensor(hand.my_action)
    
    # Concatenate all tensors
    hand_input_tensor = torch.cat([player_position, player_stack, hole_cards, prior_actions])
    
    return hand_input_tensor, my_action[:4]

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
    
def normalize(tensor):
    pass

def analyze():
    filename = r'poker_data'
    poker_hands = read_and_parse_games_from_folder(filename)

    start_moneys = []
    num_players_list = []
    player_positions = []
    player_bets = []
    player_stacks = []
    
    prior_bets = []
    prior_stacks = []
    
    for hand in poker_hands:
        try:
            start_moneys.append(hand.start_money)
            num_players_list.append(hand.num_players)
            player_positions.append(hand.player_position)
            player_bets.append(hand.my_action.player_bet)
            player_stacks.append(hand.my_action.player_stack)
            
            for actions in hand.prior_actions:
                prior_bets.append(actions.player_bet)
                prior_stacks.append(actions.player_stack)
        except:
            continue
            
    print("Min/Max start money: ", max(start_moneys), min(start_moneys))
    print("Min/Max number of players: ", max(num_players_list), min(num_players_list))
    print("Min/Max player position: ", max(player_positions), min(player_positions))
    print("Min/Max player bet: ", max(player_bets), min(player_bets))
    print("Min/Max player stack: ", max(player_stacks), min(player_stacks))
    print("Min/Max prior bet: ", max(prior_bets), min(prior_bets))
    print("Min/Max prior stack: ", max(prior_stacks), min(prior_stacks))

def normalize_scalar(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def normalize_dataset(batch_tensor):
    max_values = torch.max(batch_tensor, dim=0).values
    min_values = torch.min(batch_tensor, dim=0).values
    
    for i in range(len(batch_tensor)):
        batch_tensor[i] = torch.div(max_values - batch_tensor[i], max_values - min_values)
    
    
    

if __name__ == '__main__':
    analyze()
    filename = r'poker_data'
    poker_hands = read_and_parse_games_from_folder(filename)
    
    poker_hand_dataset = PreFlopDataset(poker_hands)
    check_hand_tensor_size(poker_hands)
    print(poker_hands[0])
    print(poker_hand_dataset[0])
    # hand = poker_hands[2]
    
    
    # max_start_money = max([hand.start_money for hand in poker_hands])
    # print(max_start_money)
