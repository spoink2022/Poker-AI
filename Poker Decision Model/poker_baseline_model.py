from poker_simulation import PokerProbabilityCalculator

def bet_size_function(win_rate, lose_rate, tie_rate, avg_tied_players, pot_size, call_size, profit):
    avg_tied_cost_rate = tie_rate / avg_tied_players
    denom = win_rate - lose_rate + avg_tied_cost_rate
    
    bet_size = (profit + (call_size - pot_size) * (win_rate + avg_tied_cost_rate)) / denom if denom != 0 else float('inf')
    bet_size_prime = 1 / denom if denom != 0 else float('inf')
    
    return bet_size, bet_size_prime


def profit_function(win_rate, lose_rate, tie_rate, avg_tied_players, pot_size, call_size, bet_size):
    avg_tied_cost_rate = tie_rate / avg_tied_players
    profit = (pot_size - call_size) * (win_rate + avg_tied_cost_rate) + bet_size * (win_rate - lose_rate + avg_tied_cost_rate)
    profit_slope = win_rate - lose_rate + avg_tied_cost_rate
    return profit, profit_slope

def get_bet_size(win_rate, lose_rate, tie_rate, avg_tied_players, pot_size, call_size, stack_size, min_raise, tight_factor=0.5):
    bet_size_root, bet_size_slope = bet_size_function(win_rate, lose_rate, tie_rate, avg_tied_players, pot_size, call_size, 0)
    call_profit, profit_slope = profit_function(win_rate, lose_rate, tie_rate, avg_tied_players, pot_size, call_size, call_size)
    min_raise_profit, _ = profit_function(win_rate, lose_rate, tie_rate, avg_tied_players, pot_size, call_size, min_raise)
    
    not_lose_variance = lose_rate * (1 - lose_rate)
    stability = 1-4*not_lose_variance # Ranges from 0 to 1
    
    rec_bet_size = stack_size*stability if profit_slope > 0 else 0
    rec_bet_size_profit, _ = profit_function(win_rate, lose_rate, tie_rate, avg_tied_players, pot_size, call_size, rec_bet_size)
    
    if stability >= max(0, min(0.9, tight_factor*1.4)) and profit_slope > 0:
        return 'allin', stack_size
    
    if stability >= tight_factor:
        if call_size > 0 and rec_bet_size > min_raise and profit_slope > 0 and rec_bet_size_profit > 0:
            return 'raises', rec_bet_size
        if call_size > 0 and call_profit > 0 and min_raise_profit < call_profit:
            return 'calls', call_size
        if call_size == 0 and profit_slope > 0 and rec_bet_size_profit > 0:
            return 'bets', rec_bet_size
        if call_size == 0 and profit_slope < 0 and call_profit < 0:
            return 'checks', 0
        
    return 'folds', 0

def baseline_model(baseline_inputs, tight_factor = 0, num_simulations=1000):
    # baseline_inputs are dataloader inputs
    
    call_size = baseline_inputs['call_size'].item()
    pot_size = baseline_inputs['pot_size'].item()
    stack_size = baseline_inputs['stack_size'].item()
    min_raise = baseline_inputs['min_raise'].item()
    hole_cards = [card[0] for card in baseline_inputs['hole_cards']]
    community_cards = [card[0] for card in baseline_inputs['community_cards']]
    num_players = baseline_inputs['num_players'].item()
    
    poker_calc = PokerProbabilityCalculator(hole_cards, community_cards, num_players, num_simulations)
    win_rate, lose_rate, tie_rate, avg_tied_players = poker_calc.poker_probability_calculator()
    
    action_name, bet_size = get_bet_size(win_rate, lose_rate, tie_rate, avg_tied_players, pot_size, call_size, stack_size, min_raise, tight_factor=tight_factor)
    
    return action_name, bet_size


    
