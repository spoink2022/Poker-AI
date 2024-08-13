from tqdm import tqdm
from torch.utils.data import DataLoader
from helper import ACTION_NAMES
import torch

# Display this in ModelInputOutputRecord
# model_label_action_name = model_io_record.action_label
# model_label_bet_size = model_io_record.bet_size_label*poker_dataset.bet_std + poker_dataset.bet_mean
# model_action_name = model_io_record.model_action_name
# model_bet_size = model_io_record.model_bet_size*poker_dataset.bet_std + poker_dataset.bet_mean

# baseline_model_inputs = baseline_io_record.baseline_inputs
# baseline_label_action_name = baseline_io_record.action_label
# baseline_label_bet_size = baseline_io_record.bet_size_label*poker_dataset.bet_std + poker_dataset.bet_mean
# baseline_action_name = baseline_io_record.baseline_action_name
# baseline_bet_size = baseline_io_record.baseline_bet_size*poker_dataset.bet_std + poker_dataset.bet_mean

# baseline_call_size = baseline_model_inputs['call_size']*poker_dataset.call_std + poker_dataset.call_mean 
# baseline_pot_size = baseline_model_inputs['pot_size']*poker_dataset.pot_std + poker_dataset.pot_mean
# baseline_stack_size = baseline_model_inputs['stack_size']*poker_dataset.stack_std + poker_dataset.stack_mean
# baseline_min_raise = baseline_model_inputs['min_raise']*poker_dataset.min_raise_std + poker_dataset.min_raise_mean
# baseline_hole_cards = baseline_model_inputs['hole_cards']
# baseline_num_players = baseline_model_inputs['num_players']

# print("Model Sample Call Size", model_call_size.item())
# print("Model Sample Pot Size", model_pot_size.item())
# print("Model Sample Stack Size", model_stack_size.item())
# print("Model Sample Min Raise", model_min_raise.item())
# print("Model Sample Hole Cards", model_hole_cards)
# print("Model Sample Num Players", model_num_players.item())

# print("\nModel Label Action Name", ACTION_NAMES[model_label_action_name.argmax().item()])
# print("Model Label Bet Size", model_label_bet_size.item())
# print("\nModel Action Name", model_action_name)
# print("Model Bet Size", model_bet_size.item())

# print("\nBaseline Sample Call Size", baseline_call_size.item())
# print("Baseline Sample Pot Size", baseline_pot_size.item())
# print("Baseline Sample Stack Size", baseline_stack_size.item())
# print("Baseline Sample Min Raise", baseline_min_raise.item())
# print("Baseline Sample Hole Cards", baseline_hole_cards)
# print("Baseline Sample Num Players", baseline_num_players.item())

# print("\nBaseline Label Action Name", ACTION_NAMES[baseline_label_action_name.argmax().item()])
# print("Baseline Label Bet Size", baseline_label_bet_size.item())
# print("\nBaseline Action Name", baseline_action_name)
# print("Baseline Bet Size", baseline_bet_size.item())

class ModelInputOutputRecord:
    def __init__ (self, input_tensor, model_action_name, model_bet_size, baseline_action_name, baseline_bet_size, action_label, bet_size_label, baseline_inputs, game_idx):
        self.input_tensor = input_tensor
        self.model_action_name = model_action_name
        self.model_bet_size = model_bet_size
        self.baseline_action_name = baseline_action_name
        self.baseline_bet_size = baseline_bet_size
        self.action_label = action_label
        self.bet_size_label = bet_size_label
        self.baseline_inputs = baseline_inputs
        self.game_idx = game_idx
        
    def display(self, dataset):
        game = dataset.games[self.game_idx]
        
        inputs = self.baseline_inputs
        call_size = (inputs['call_size']*dataset.call_std + dataset.call_mean).item()
        pot_size = (inputs['pot_size']*dataset.pot_std + dataset.pot_mean).item()
        stack_size = (inputs['stack_size']*dataset.stack_std + dataset.stack_mean).item()
        min_raise = (inputs['min_raise']*dataset.min_raise_std + dataset.min_raise_mean).item()
        hole_cards = [card[0] for card in inputs['hole_cards']]
        num_players = inputs['num_players'].item()
        
        label_action_name = ACTION_NAMES[self.action_label.argmax().item()]
        label_bet_size = (self.bet_size_label*dataset.bet_std + dataset.bet_mean).item()
        
        model_output_action = self.model_action_name
        model_output_bet_size = (self.model_bet_size*dataset.bet_std + dataset.bet_mean).item()
        
        baseline_output_action = self.baseline_action_name
        baseline_output_bet_size = self.baseline_bet_size
        
        print('Dataset Game Index', self.game_idx)
        print(game)
        
        print("------------------------------------")
        print("\nModel Sample Call Size", call_size)
        print("Model Sample Pot Size", pot_size)
        print("Model Sample Stack Size", stack_size)
        print("Model Sample Min Raise", min_raise)
        print("Model Sample Hole Cards", hole_cards)
        print("Model Sample Number of Players", num_players)
        
        print("\nAccuracy comparison")
        print('Action Name\n' "Label:", label_action_name, "\tModel Output:", model_output_action, "\tBaseline Output:", baseline_output_action)
        print('Bet Size\n' "Label:", round(label_bet_size, 2), "\tModel Output:", round(model_output_bet_size, 2), "\tBaseline Output:", round(baseline_output_bet_size, 2))
        
        
        
        
        

def get_models_accuracy(model, baseline_model, dataset):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model_tracker = {action_name: {'correct': [], 'wrong': []} for action_name in ACTION_NAMES}
    baseline_tracker = {action_name: {'correct': [], 'wrong': []} for action_name in ACTION_NAMES}
    
    for input_tensor, action_label, bet_size_label, baseline_inputs, game_idx in tqdm(dataloader, desc='Evaluate', leave=False):        
        model_action, model_bet_size = model(input_tensor)
        baseline_action_name, baseline_bet_size = baseline_model(baseline_inputs)
        
        model_action_name = ACTION_NAMES[torch.argmax(model_action, dim=1).item()]
        label_action_name = ACTION_NAMES[torch.argmax(action_label, dim=1).item()]
        
        record = ModelInputOutputRecord(input_tensor, model_action_name, model_bet_size, baseline_action_name, \
            baseline_bet_size, action_label, bet_size_label, baseline_inputs, game_idx)
        
        if model_action_name == label_action_name:
            model_tracker[model_action_name]['correct'].append(record)
        else:
            model_tracker[model_action_name]['wrong'].append(record)
            
        if baseline_action_name == label_action_name:
            baseline_tracker[baseline_action_name]['correct'].append(record)
        else:
            baseline_tracker[baseline_action_name]['wrong'].append(record)
            
    return model_tracker, baseline_tracker
    
        
        
        
        

    