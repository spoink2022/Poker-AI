import os
import re

import torch
from torch.utils.data import Dataset
import helper as hp
from itertools import chain
import numpy as np
from tqdm import tqdm

from helper import ACTION_NAMES, ROUND_NAMES, PokerPlayer
import operator

def rearrange_list(original_list, indices):
    if len(original_list) != len(indices):
        raise ValueError("The length of the original list and indices list must be the same")
    
    if set(indices) != set(range(len(indices))):
        raise ValueError("Invalid indices. Each index must appear exactly once")
    
    return list(map(original_list.__getitem__, indices))

class PokerState():
    def __init__(self, community_cards, pot_size, round_call_amount, player_round_bet_size, player_stack):
        self.community_cards = community_cards
        self.pot_size = pot_size
        self.round_call_amount = round_call_amount
        self.player_round_bet_size = player_round_bet_size
        self.player_stack = player_stack
        
    def __repr__(self) -> str:
        return f"Community Cards: {self.community_cards}\nPot Size: {self.pot_size}\nRound Call Amount: {self.round_call_amount}\nPlayer Round Bet Size: {self.player_round_bet_size}\nPlayer Stack: {self.player_stack}\n"

class PokerRound():
    def __init__(self, pti, players, round_data, pot_size):
        self.pti = pti # Player to Index
        self.players = players
        self.round_data = round_data
        self.round_info = []
        self.pot_size = pot_size
        self.round_call_amount = 0
        
        if round_data["name"] in ROUND_NAMES:
            self.process_round(round_data)
        else:
            raise ValueError("Invalid round name")
        
    def process_round(self, round_data):
        round_name = round_data["name"]
        for line in round_data["lines"]:            
            # Extract the player name and action using regular expressions
            match = re.match(r'Player (.*) (\w+)', line)
            number_match = re.search(r'\((\d+(\.\d+)?)\)$', line)
            out_match = re.match(r"Player (.*) is timed out", line)
            
            if out_match:
                player_name = out_match.group(1)
                action_name = "timed out"
                bet_size = 0
                
                player = self.players[self.pti[player_name]]
                
                poker_state = None
                
                player.insert_round_action(round_name, (action_name, bet_size, poker_state))
                self.round_info.append((player_name, action_name, bet_size, poker_state))
                
            elif match: 
                player_name = match.group(1)
                action_name = match.group(2)
                bet_size = float(number_match.group(1)) if number_match else 0
                
                player = self.players[self.pti[player_name]]
                
                poker_state = PokerState([], self.pot_size, self.round_call_amount, player.round_bet_size[round_name], player.stack)
                
                player.insert_round_action(round_name, (action_name, bet_size, poker_state))
                self.round_info.append((player_name, action_name, bet_size, poker_state))
                
                self.pot_size += float(bet_size)
                self.round_call_amount = player.round_bet_size[round_name] if player.round_bet_size[round_name] > self.round_call_amount else self.round_call_amount
                
    def display_round(self):
        display = f"{self.round_data['name']}:\n"
        for player_name, action_name, bet_size, poker_state in self.round_info:
            if bet_size is None:
                display += f"{player_name} {action_name}\n"
            else:
                display += f"{player_name} {action_name} {bet_size}"
        display += "\n"
        return display
    
    def __repr__(self):
        display = f"{self.round_data['name']}:\n"
        for player_name, action_name, bet_size, poker_state in self.round_info:
            if bet_size is None:
                display += f"{player_name} {action_name}\n"
            else:
                display += f"{player_name} {action_name} {bet_size} \n{poker_state}\n"
        display += "\n"
        return display
    
class PokerGame():
    def __init__(self, game_data):
        self.stp = {} # Seat to player
        self.pti = {} # Player to index
        self.rounds = []
        self.players = []
        self.button_seat = None
        self.game_data = game_data
        self.community_cards = {'FLOP': [], 'TURN': [], 'RIVER': []}
        self.pot_size = 0 # Keeps track of pot size
        self.round_call_amount = 0 # Keeps track of the amount required to call
        self.process_game()
    
    def process_game(self):
        # Extract button seat
        for i, line in enumerate(self.game_data):
            match = re.match(r"Seat (\d+) is the button", line)
            if match:
                self.button_seat = int(match.group(1))
            if re.match(r"Seat (\d+): (.*) \(([\d.]+)\)\.", line):
                break
        
        # Extract player data
        for i, line in enumerate(self.game_data[i:], i):
            match = re.match(r"Seat (\d+): (.*) \(([\d.]+)\)\.", line)
            
            if match:
                player_seat = int(match.group(1))
                player_name = match.group(2)
                player_stack = float(match.group(3))
                
                poker_player = PokerPlayer(player_name, player_seat, player_stack)
                self.stp[player_seat] = poker_player
                self.players.append(poker_player)
            else:
                break
        
        # Rearrange player list according to the button. 
        self.pti = {player.name: i for i, player in enumerate(self.players)}
        button_player_index = self.pti[self.stp[self.button_seat].name]
        self.players = rearrange_list(self.players, list(range(button_player_index+1, len(self.players))) + list(range(button_player_index+1)))
        
        # Create player to index mapping
        self.pti = {player.name: i for i, player in enumerate(self.players)}
            
        # Initiate blinds
        for i, line in enumerate(self.game_data[i:], i):
            match = re.match(r"Player (.*) has (\w+) blind \(([\d.]+)\)", line)
            straddle_match = re.match(r"Player (.*) straddle \(([\d.]+)\)", line)
            post_match = re.match(r"Player (.*) posts \(([\d.]+)\)", line)
            
            out_match = re.match(r"Player (.*) (sitting out|is timed out|wait BB)", line)
            
            if match:
                player_name = match.group(1)
                player_blind = match.group(2)
                player_bet_size = float(match.group(3))
                action_name = player_blind
                
                player = self.players[self.pti[player_name]]
                
                poker_state = PokerState([], self.pot_size, self.round_call_amount, player.round_bet_size["PREFLOP"], player.stack)
                player.insert_round_action("PREFLOP", (action_name, player_bet_size, poker_state))
                
                self.round_call_amount = player.round_bet_size['PREFLOP'] if player.round_bet_size['PREFLOP'] > self.round_call_amount else self.round_call_amount
                self.pot_size += float(player_bet_size)
                
            elif straddle_match:
                player_name = straddle_match.group(1)
                player_bet_size = float(straddle_match.group(2))
                action_name = "straddle"
                
                player = self.players[self.pti[player_name]]
                
                poker_state = PokerState([], self.pot_size, self.round_call_amount, player.round_bet_size['PREFLOP'], player.stack)
                player.insert_round_action("PREFLOP", (action_name, player_bet_size, poker_state))
                
                self.round_call_amount = player.round_bet_size['PREFLOP'] if player.round_bet_size['PREFLOP'] > self.round_call_amount else self.round_call_amount
                self.pot_size += float(player_bet_size)
                
            elif post_match:
                player_name = post_match.group(1)
                player_bet_size = float(post_match.group(2))
                action_name = "posts"
                
                player = self.players[self.pti[player_name]]
                
                poker_state = PokerState([], self.pot_size, self.round_call_amount, player.round_bet_size['PREFLOP'], player.stack)
                player.insert_round_action("PREFLOP", (action_name, player_bet_size, poker_state))
                
                self.round_call_amount = player.round_bet_size['PREFLOP'] if player.round_bet_size['PREFLOP'] > self.round_call_amount else self.round_call_amount
                self.pot_size += float(player_bet_size)
            
            # Remove player if they are sitting out
            elif out_match:
                player_name = out_match.group(1)
                if player_name in self.pti:
                    player_index = self.pti[player_name]
                else:
                    continue
                self.players.pop(player_index)
                self.pti = {player.name: i for i, player in enumerate(self.players)}
            else:
                break
        
        # Extract player card data
        for i, line in enumerate(self.game_data[i:], i):
            match = re.match(r"Player (.*) received a card\.", line)
            if match:
                pass
            else:
                break
        for i, line in enumerate(self.game_data[i:], i):
            match = re.match(r"Player (.*) received card: \[(.*?)\]", line)
            if match:
                player_name = match.group(1)
                player_hole_cards = match.group(2)
                self.players[self.pti[player_name]].hole_cards.append(player_hole_cards)
            else:
                break
        for i, line in enumerate(self.game_data[i:], i):
            match = re.match(r"Player (.*) received a card\.", line)
            if match:
                pass
            else:
                break
            
        # Extract the FLOP data
        rounds_data = {"PREFLOP": [], "FLOP": [], "TURN": [], "RIVER": []}
        curr_round_name = "PREFLOP"
        for i, line in enumerate(self.game_data[i:], i):
            end_match = re.match(r"(------ Summary ------|Uncalled bet \(([\d.]+)\) returned to (.*)|Player (.*) mucks cards)", line)
            match = re.match(r'\*\*\* (FLOP|TURN|RIVER) \*\*\*: \[(.*?)\](?:\s+\[(.*?)\])?', line)
            
            # print(match)
            # print(line)
            
            if end_match:
                break
            
            if match:
                curr_round_name = match.group(1) if match.group(1) in ROUND_NAMES else None
                self.community_cards[curr_round_name] = match.group(2).split()
                if match.group(3):
                    self.community_cards[curr_round_name] += match.group(3).split()        
                continue
            
            if curr_round_name in ROUND_NAMES:
                rounds_data[curr_round_name].append(line)
                
            
        for round_name, round_lines in rounds_data.items():
            poker_round = PokerRound(self.pti, self.players, {"name": round_name, "lines": round_lines}, self.pot_size)
            self.pot_size = poker_round.pot_size
            self.rounds.append(poker_round)
        
    def __repr__(self):
        display = "--------------------------------\n"
        display += "Players:\n"
        for player in self.players:
            display += player.name+'\n'
            
        display += f"\nButton Player: {self.players[-1].name}\n\n"
        
        for rounds in self.rounds:
            display += str(rounds)
        return display

class PokerDataset(Dataset):
    def __init__(self, data_dir, main_player_name, transform = None, balance_data = False, include_actions = ['calls', 'raises', 'bets', 'folds', 'checks', 'allin']):
        self.data_dir = data_dir
        self.main_player_name = main_player_name
        self.transform = transform
        self.games_data = []
        self.games = []
        self.read_games()
        self.parse_games()
        
        suits = ['s', 'h', 'd', 'c']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.identity = torch.eye(52)
        self.cards = [rank+suit for suit in suits for rank in ranks]
        self.max_pos = self._get_max_pos()
        self._get_mean_std_features()
        
        self.output_indices = {action_name: [] for action_name in include_actions}
        self.processed_outputs = []
        
        index = 0
        for i in tqdm(range(len(self.games)), desc="Processing Items", leave=True):
            input_tensor, label_action_tensor, label_bet_size_tensor, baseline_inputs = self.process_item(i)
            action_name = ACTION_NAMES[label_action_tensor.argmax().item()]
            
            if action_name not in include_actions:
                continue
            
            output_datum = (input_tensor, label_action_tensor, label_bet_size_tensor, baseline_inputs, i)
            self.processed_outputs.append(output_datum)
            
            self.output_indices[action_name].append(index)
            index += 1
            
        # Balance the data for each action type unless 0 indices in output_indices
        # Randomly repeats indices uniformly for each acttion type to the largest action type length
        if balance_data:
            self._balance_data()
            return 
        
        self.balanced_indices = []
        for indices in self.output_indices.values():
            self.balanced_indices += indices
            
    def _balance_data(self):
        max_length = max(len(indices) for indices in self.output_indices.values())
        for action_name, indices in self.output_indices.items():
            if len(indices) > 0:
                multiple = max_length // len(indices)
                self.output_indices[action_name] = indices*multiple
                remainder = max_length % len(indices)
                self.output_indices[action_name] += np.random.choice(indices, remainder, replace=False).tolist()
        
        self.balanced_indices = []
        for indices in self.output_indices.values():
            self.balanced_indices += indices
        
    def __len__(self):
        return len(self.balanced_indices)

    def read_games(self):
        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, "r") as file:
                for line in file:
                    if line.startswith("Game started at:"):
                        #print("Game started\n\n")
                        self.games_data.append([line])
                    else:
                        self.games_data[-1].append(line)

    def parse_games(self):
        self.action_count = {}
        self.action_count['timed out'] = 0
        for action_name in ACTION_NAMES:
            self.action_count[action_name] = 0
            
        for game_data in tqdm(self.games_data, desc="Parsing Games", leave=True):
            poker_game = PokerGame(game_data)
            if self.main_player_name not in poker_game.players:
                continue
            main_player = poker_game.players[poker_game.pti[self.main_player_name]]
            if len(main_player.hole_cards) == 0:
                continue
            if len(main_player.actions["PREFLOP"]) == 0:
                continue
            if len(main_player.actions["PREFLOP"]) == 1 and main_player.actions["PREFLOP"][0][0] == 'big':
                continue 
            for round in poker_game.rounds:
                for action in round.round_info:
                    action_name = action[1]
                    self.action_count[action_name] += 1
            self.games.append(poker_game)
            
    def _get_max_pos(self):
        return max([len(game.players) for game in self.games])
    
    def _get_mean_std_features(self):
        all_stacks = []
        all_bets = []
        all_call_sizes = []
        all_pot_sizes = []
        all_min_raises = []
        for game in self.games:
            game_stacks = torch.tensor([player.stack for player in game.players])
            all_stacks.append(game_stacks)
            for round in game.rounds:
                for player_name, action_name, bet_size, poker_state in round.round_info:
                    if bet_size is not None and bet_size != 0:
                        all_bets.append(bet_size)
                    if poker_state is not None:
                        all_call_sizes.append(poker_state.round_call_amount - poker_state.player_round_bet_size)
                        all_pot_sizes.append(poker_state.pot_size)
                        all_min_raises.append(2*poker_state.round_call_amount)
                        
        all_bets = torch.tensor(all_bets)
        all_stacks = torch.cat(all_stacks, dim=0)
        all_call_sizes = torch.tensor(all_call_sizes)
        all_pot_sizes = torch.tensor(all_pot_sizes)
        all_min_raises = torch.tensor(all_min_raises)
        
        self.call_mean, self.call_std = all_call_sizes.mean(), all_call_sizes.std()
        self.pot_mean, self.pot_std = all_pot_sizes.mean(), all_pot_sizes.std()
        self.min_raise_mean, self.min_raise_std = all_min_raises.mean(), all_min_raises.std()
        self.stack_mean, self.stack_std = all_stacks.mean(), all_stacks.std()
        self.bet_mean, self.bet_std = all_bets.mean(), all_bets.std()
    
    def convert_card_to_tensor(self, card):
        card_idx = self.cards.index(card)
        card_tensor = self.identity[card_idx]
        return card_tensor
            
    def get_player_pos_tensor(self, player, game_idx):
        player_pos = self.games[game_idx].players.index(player)
        player_pos /= self.max_pos
        return torch.tensor(player_pos).unsqueeze(0)
    
    def get_hole_cards_tensor(self, player, game_idx):
        player_idx = self.games[game_idx].players.index(player)
        hole_cards = self.games[game_idx].players[player_idx].hole_cards
        card_tensors = [self.convert_card_to_tensor(card) for card in hole_cards]
        if len(card_tensors) == 0:
            print(self.games[game_idx])
        return torch.cat(card_tensors, dim=0)
    
    def get_community_cards_tensor(self, round_name, game_idx):
        community_cards = self.games[game_idx].community_cards[round_name]
        card_tensors = [self.convert_card_to_tensor(card) for card in community_cards]
        cards_tensor = torch.cat(card_tensors, dim=0)
        fixed_length = 52*5
        pad_length = fixed_length - len(cards_tensor)
        
        if pad_length > 0:
            cards_tensor = torch.nn.functional.pad(cards_tensor, (0, pad_length), mode="constant", value=0)
            
        return cards_tensor
    
    def get_past_actions_tensor(self, round_name, main_player_name, game_idx):
        game = self.games[game_idx]
        round_idx = ROUND_NAMES.index(round_name)
        round_info = game.rounds[round_idx].round_info
        player_idx = game.pti[main_player_name]
        num_players = len(game.players)
        
        baseline_inputs = {'call_size': 0, 'pot_size': 0, 'stack_size': 0, 'min_raise': 0, 'hole_cards': 0, 'community_cards': 0, 'num_players': 0}
        num_actions = len(round_info)
        past_actions_tensor = []
        
        # Includes action_tensor (11), stack_size (1), bet_size (1), call_size (1), pot_size (1), stack_size (1), min_raise (1), num_players (1)
        action_tensor_size = len(ACTION_NAMES) + 6
        
        i = 1
        main_player_action_tensor = None
        main_player_input_tensor = None
        while len(past_actions_tensor) < num_players:
            if num_actions < i:
                break
            player_name, action_name, bet_size, player_state = round_info[-i]
            i += 1
            bet_size = float(bet_size) if bet_size is not None else 0
            if player_name == main_player_name:
                if action_name == "timed out": # Skip timed out actions
                    continue
                
                call_size = player_state.round_call_amount - player_state.player_round_bet_size
                pot_size = player_state.pot_size
                stack_size = player_state.player_stack
                min_raise = 2*player_state.round_call_amount
                num_players_norm = len(game.players)/self.max_pos
                
                call_size_tensor = ((call_size - self.call_mean) / self.call_std).unsqueeze(0)
                pot_size_tensor = ((pot_size - self.pot_mean) / self.pot_std).unsqueeze(0)
                stack_size_tensor = ((stack_size - self.stack_mean) / self.stack_std).unsqueeze(0)
                min_raise_tensor = ((min_raise - self.min_raise_mean) / self.min_raise_std).unsqueeze(0)
                num_players_tensor = torch.tensor(num_players_norm).unsqueeze(0)
                
                action_idx = ACTION_NAMES.index(action_name)
                action_tensor = self.identity[action_idx][:len(ACTION_NAMES)]
                norm_bet_size = ((bet_size - self.bet_mean) / self.bet_std).unsqueeze(0)
                main_player_action_tensor = torch.cat([action_tensor, norm_bet_size], dim=0)
                main_player_input_tensor = torch.cat([call_size_tensor, pot_size_tensor, stack_size_tensor, min_raise_tensor, num_players_tensor], dim=0)
                
                baseline_inputs['call_size'] = call_size
                baseline_inputs['pot_size'] = pot_size
                baseline_inputs['stack_size'] = stack_size
                baseline_inputs['min_raise'] = min_raise
                baseline_inputs['hole_cards'] = game.players[game.pti[player_name]].hole_cards
                baseline_inputs['community_cards'] = game.community_cards[round_name] if round_name in game.community_cards else []
                baseline_inputs['num_players'] = len(game.players)
                continue
            if main_player_action_tensor == None:
                continue
            if action_name == "timed out":
                continue
            
            call_size = player_state.round_call_amount - player_state.player_round_bet_size
            pot_size = player_state.pot_size
            stack_size = player_state.player_stack
            min_raise = 2*player_state.round_call_amount
            num_players_norm = len(game.players)/self.max_pos
            
            call_size_tensor = ((call_size - self.call_mean) / self.call_std).unsqueeze(0)
            pot_size_tensor = ((pot_size - self.pot_mean) / self.pot_std).unsqueeze(0)
            stack_size_tensor = ((stack_size - self.stack_mean) / self.stack_std).unsqueeze(0)
            min_raise_tensor = ((min_raise - self.min_raise_mean) / self.min_raise_std).unsqueeze(0)
            num_players_tensor = torch.tensor(num_players_norm).unsqueeze(0)
            
            action_idx = ACTION_NAMES.index(action_name)
            action_tensor = self.identity[action_idx][:len(ACTION_NAMES)]
            player = game.players[game.pti[player_name]]
            player_norm_stack = ((player.stack - self.stack_mean) / self.stack_std).unsqueeze(0)
            norm_bet_size = ((bet_size - self.bet_mean) / self.bet_std).unsqueeze(0)
            past_action_tensor = torch.cat([action_tensor, player_norm_stack, norm_bet_size, call_size_tensor, pot_size_tensor, stack_size_tensor, min_raise_tensor], dim=0)
            #past_action_tensor = torch.cat([action_tensor, player_norm_stack, norm_bet_size], dim=0)
            past_actions_tensor.append(past_action_tensor)
        
        past_actions_tensor = past_actions_tensor if past_actions_tensor else [torch.zeros(size=(1,))]
        #length, sizes = len(past_actions_tensor), [action_tensor.shape for action_tensor in past_actions_tensor]
        
        past_actions_tensor = torch.cat(past_actions_tensor, dim=0)
        fixed_length = self.max_pos*action_tensor_size
        
        pad_length = fixed_length - len(past_actions_tensor)
        
        if pad_length > 0:
            past_actions_tensor = torch.nn.functional.pad(past_actions_tensor, (0, pad_length), mode="constant", value=0)
            
        if main_player_action_tensor is None:
            print(self.games[game_idx])
            
        # if pad_length < 0:
        #     print(length, sizes)
        #     print(past_actions_tensor.shape)
        #     print(game)
        
        return past_actions_tensor, main_player_action_tensor, baseline_inputs, main_player_input_tensor
    
    def process_item(self, index):
        player_pos = self.get_player_pos_tensor(self.main_player_name, index)
        hole_cards = self.get_hole_cards_tensor(self.main_player_name, index)
        #community_cards = self.get_community_cards_tensor("PREFLOP", index)
        main_player_index = round(player_pos.item()*self.max_pos)
        stack = self.games[index].players[main_player_index].stack
        stack = ((stack - self.stack_mean) / self.stack_std).unsqueeze(0)
        past_actions_tensor, main_player_action_tensor, baseline_inputs, main_player_input_tensor = self.get_past_actions_tensor("PREFLOP", self.main_player_name, index)
        #return player_pos, hole_cards, stack, past_actions_tensor, main_player_action_tensor
        input_tensor = torch.cat([player_pos, hole_cards, stack, main_player_input_tensor, past_actions_tensor], dim=0)
        label_action_tensor = main_player_action_tensor[:-1]
        label_bet_size_tensor = main_player_action_tensor[-1:]
        return input_tensor, label_action_tensor, label_bet_size_tensor, baseline_inputs
    
    def __getitem__(self, index):
        b_i = self.balanced_indices[index]
        input_tensor, label_action_tensor, label_bet_size_tensor, baseline_inputs, game_idx = self.processed_outputs[b_i]
        return input_tensor, label_action_tensor, label_bet_size_tensor, baseline_inputs, game_idx
        
    def _test_function(self):
        # see how many available options there are
        
        # action_name
        all_action_name = []
        all_bet_sizes = []
        for game in self.games:
            for round in game.rounds:
                for player_name, action_name, bet_size in round.round_info:
                    all_action_name.append(action_name)
                    bet_value = float(bet_size) if bet_size is not None else 0
                    all_bet_sizes.append(bet_value)
        print(set(all_action_name))
        all_bet_sizes = np.array(all_bet_sizes, dtype=float)
        hp.plot_distribution(all_bet_sizes)
        # remove 0s
        all_bet_sizes = all_bet_sizes[all_bet_sizes != 0]
        hp.plot_distribution(all_bet_sizes)
        
                

    
    
        
        


if __name__ == "__main__":
    filepath = r'C:\Users\kctak\Documents\Code\PokerAI\APS360 Progress\poker_data'
    poker_dataset = PokerDataset(filepath, "IlxxxlI")
    print(len(poker_dataset))
    print({action_name: len(indices) for action_name, indices in poker_dataset.output_indices.items()})
    #print(poker_dataset[0])
    poker_dataset._balance_data()
    print(len(poker_dataset))
    print({action_name: len(indices) for action_name, indices in poker_dataset.output_indices.items()})
    # print(poker_dataset[0])
    # test = []
    # for i in tqdm(range(len(poker_dataset))):
    #     input_shape, action_shape, bet_shape = poker_dataset[i][0].shape, poker_dataset[i][1].shape, poker_dataset[i][2].shape
    #     test.append((input_shape, action_shape, bet_shape))
    # print(set(test))
    #poker_dataset._test_function()
    
    # print(poker_dataset.get_player_pos_tensor("IlxxxlI", 0))
    # print(poker_dataset.cards)
    # print(poker_dataset.get_hole_cards_tensor("IlxxxlI", 0))
    
    # print(poker_dataset.games[1].community_cards)
    # com_cards = poker_dataset.get_community_cards_tensor("FLOP", 1)
    # print(com_cards.shape)
    # print(com_cards)
    
    
    