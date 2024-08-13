from tqdm import tqdm
from torch.utils.data import DataLoader
from helper import ACTION_NAMES
import torch

class ModelInputOutputRecord:
    def __init__ (self, input_tensor, model_action_name, model_bet_size, baseline_action_name, baseline_bet_size, action_label, bet_size_label, baseline_inputs):
        self.input_tensor = input_tensor
        self.model_action_name = model_action_name
        self.model_bet_size = model_bet_size
        self.baseline_action_name = baseline_action_name
        self.baseline_bet_size = baseline_bet_size
        self.action_label = action_label
        self.bet_size_label = bet_size_label
        self.baseline_inputs = baseline_inputs
        

def get_models_accuracy(model, baseline_model, dataset):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model_tracker = {action_name: {'correct': [], 'wrong': []} for action_name in ACTION_NAMES}
    baseline_tracker = {action_name: {'correct': [], 'wrong': []} for action_name in ACTION_NAMES}
    
    for input_tensor, action_label, bet_size_label, baseline_inputs in tqdm(dataloader, desc='Evaluate', leave=False):        
        model_action, model_bet_size = model(input_tensor)
        baseline_action_name, baseline_bet_size = baseline_model(baseline_inputs)
        
        model_action_name = ACTION_NAMES[torch.argmax(model_action, dim=1).item()]
        label_action_name = ACTION_NAMES[torch.argmax(action_label, dim=1).item()]
        
        record = ModelInputOutputRecord(input_tensor, model_action_name, model_bet_size, baseline_action_name, \
            baseline_bet_size, action_label, bet_size_label, baseline_inputs)
        
        if model_action_name == label_action_name:
            model_tracker[model_action_name]['correct'].append(record)
        else:
            model_tracker[model_action_name]['wrong'].append(record)
            
        if baseline_action_name == label_action_name:
            baseline_tracker[baseline_action_name]['correct'].append(record)
        else:
            baseline_tracker[baseline_action_name]['wrong'].append(record)
            
    return model_tracker, baseline_tracker
    
        
        
        
        

    