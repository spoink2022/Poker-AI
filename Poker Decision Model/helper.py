import matplotlib.pyplot as plt
import seaborn as sns
from poker_hand_evaluation import HandEvaluation, Card

SUITS = ['s', 'h', 'd', 'c']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
CARDS = [rank+suit for suit in SUITS for rank in RANKS]

ROUND_NAMES = ["PREFLOP", "FLOP", "TURN", "RIVER"]
ACTION_NAMES = ['small', 'big', 'calls', 'caps', 'raises', 'bets', 'folds', 'posts', 'checks', 'straddle', 'allin']

def check_valid_card_str(card_str):
    if len(card_str) == 2 and card_str[0] in RANKS and card_str[1] in SUITS:
        return True
    if len(card_str) == 3 and card_str[:2] in RANKS and card_str[2] in SUITS:
        return True
    return False
        
def convert_to_card(card_str):
    if not check_valid_card_str(card_str):
        print(card_str)
        raise ValueError("Invalid card string")
    
    if len(card_str) == 3:
        rank = 10
        suit = card_str[2]
        return Card(rank, suit)
    
    rank = int(card_str[0]) if card_str[0] in RANKS[:9] else 10+RANKS.index(card_str[0])
    suit = card_str[1]
    return Card(rank, suit)

def evaluate_hand(hole_cards, community_cards):
    list_of_cards = [convert_to_card(card_str) for card_str in hole_cards + community_cards]
    hand_val = HandEvaluation(list_of_cards)
    return hand_val

class PokerPlayer():
    def __init__(self, name, seat, stack, hole_cards = None, actions = None, action_func = None):
        self.name = name
        self.seat = seat
        self.stack = stack # Make stack into list to keep track of stack changes
        self.action_func = action_func if isinstance(action_func, type(lambda x: x)) else None
        self.round_bet_size = {round_name: 0 for round_name in ROUND_NAMES}

        if isinstance(hole_cards, tuple) and len(hole_cards) == 2 and all([isinstance(card, str) for card in hole_cards]):
            self.hole_cards = hole_cards
        elif hole_cards is None:
            self.hole_cards = []
        else:
            raise ValueError("Invalid hole cards")
        
        if actions is None:
            self.actions = {round_name: [] for round_name in ROUND_NAMES}
        elif isinstance(actions, dict) and all([isinstance(keys, str) and all([isinstance(action_name, str) and isinstance(bet_size, float) \
            for action_name, bet_size in values]) for keys, values in actions.items()]):
            self.actions = actions
        else:
            raise ValueError("Invalid actions")
        
    def get_round_action(self, round_name, cycle_num):
        if isinstance(cycle_num, int) and cycle_num < 1 and cycle_num > len(self.actions[round_name]):
            raise ValueError("Invalid cycle number, must be at least 1 or in bounds")
        return self.actions[round_name][cycle_num-1]
    
    def insert_round_action(self, round_name, action, cycle_num = None):
        if cycle_num is None:
            self.actions[round_name].append(action)
            self.round_bet_size[round_name] += action[1]
            self.stack -= action[1]
        if isinstance(cycle_num, int) and cycle_num < 1:
            raise ValueError("Invalid cycle number, must be at least 1")
        if isinstance(cycle_num, int):
            action_name, bet_size = action
            self.actions[round_name].insert(cycle_num-1, (action_name, bet_size))
            
    def get_action_for_game(self, input_data):
        if self.action_func is None:
            raise ValueError("No action function provided")
        action, bet_size = self.action_func(input_data)
        return action, bet_size
        
    def __repr__(self):
        return f"Player: {self.name}, Stack: {self.stack}, Actions: {self.actions}, Hole Cards: {self.hole_cards}.\n"

    def __eq__(self, other):
        if hasattr(other, 'name'):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        
        return self.name == other
    
    def __hash__(self):
        return hash(self.name)

def plot_distribution(data):
    """
    Plots the distribution of a list of floats.
    
    Parameters:
    data (list of floats): The data to plot.
    """
    # Check if the input data is a list of floats
    if not all(isinstance(i, float) for i in data):
        raise ValueError("Input data should be a list of floats.")
    
    # Set the style for the plot
    sns.set_theme(style="whitegrid")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the distribution using seaborn
    sns.histplot(data, kde=True, bins=30)
    
    # Set plot labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Data')
    
    # Show the plot
    plt.show()

