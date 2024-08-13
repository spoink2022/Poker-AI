from helper import PokerPlayer, CARDS, ROUND_NAMES, ACTION_NAMES, evaluate_hand
import numpy as np

class TexasHoldemPokerGame():
    def __init__(self, players: list[PokerPlayer], small_blind, big_blind):
        # Initialize the poker game with players, small blind, and big blind
        # Players must be in order from small_blind to button
        self.deck = CARDS.copy()
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.pot_size = 0
        self.community_cards = []
        
    def shuffle_deck(self):
        # Shuffles the deck
        np.random.shuffle(self.deck)
        
    def deal_hole_cards(self, indices: list[int] = None):
        # Deals cards to each player according to the player indices
        indices = indices if indices is not None else range(len(self.players))
        for i in indices:
            self.players[i].hole_cards = [self.deck.pop(), self.deck.pop()]
    
    def deal_community_cards(self, num_cards):
        # Deals community cards
        for i in range(num_cards):
            self.community_cards.append(self.deck.pop())
            
    def print_game_state(self, curr_call_size = None): 
        # Prints the game state
        print(f"Pot Size: {self.pot_size}")
        print("Community Cards: ", self.community_cards)
        for player in self.players:
            print(player)
        if curr_call_size:
            print(f"Current Call Size: {curr_call_size}")
            
    def play_round(self, round_name):
        # Proceed with the preflop round
        if round_name not in ROUND_NAMES:
            raise ValueError("Invalid round name")
        if round_name == "FLOP":
            self.deal_community_cards(3)
        elif round_name == "TURN" or round_name == "RIVER":
            self.deal_community_cards(1)
        
        i = 0
        player_index = 0
        aggressor_index = None
        curr_call_size = 0
        allin_players = []
        while True:
            player_index = i % len(self.players)
            
            if aggressor_index == player_index:
                break
            
            if player_index in allin_players:
                i += 1
                continue
                
            curr_player = self.players[player_index]
            
            if i == 0 and round_name == "PREFLOP":
                aggressor_index = player_index
                
                curr_player.stack -= self.small_blind
                curr_player.insert_round_action(round_name, ("small", self.small_blind))
                self.pot_size += self.small_blind
                curr_call_size = self.small_blind
                i += 1
                continue
            elif i == 1 and round_name == "PREFLOP":
                aggressor_index = player_index
                
                curr_player.stack -= self.big_blind
                curr_player.insert_round_action(round_name, ("big", self.big_blind))
                self.pot_size += self.big_blind
                curr_call_size = self.big_blind
                i += 1
                continue
            elif i == 0:
                aggressor_index = player_index
            
            self.print_game_state(curr_call_size)
            action, bet_size = curr_player.get_action_for_game(None)
            player_round_bet_size = curr_player.round_bet_size[round_name]
            
            if action == "folds":
                self.players.remove(curr_player)
                continue
            elif action == "checks":
                curr_player.insert_round_action(round_name, (action, 0))
            elif action == "calls":
                curr_player.stack -= curr_call_size - player_round_bet_size
                curr_player.insert_round_action(round_name, (action, curr_call_size - player_round_bet_size))
                self.pot_size += curr_call_size - player_round_bet_size
            elif action == "bets":
                if curr_call_size > 0:
                    raise ValueError("Cannot bet when there is a current call size")
                aggressor_index = player_index
                
                curr_player.stack -= bet_size
                curr_player.insert_round_action(round_name, (action, bet_size))
                self.pot_size += bet_size
                curr_call_size = bet_size
            elif action == "raises":
                if round((player_round_bet_size+bet_size)*100) < 2*round(100*curr_call_size):
                    raise ValueError(f"Raise must be greater than twice the current call size and \
                        requires at least bet_size of {2*curr_call_size - player_round_bet_size} but got {bet_size}")
                aggressor_index = player_index
                
                curr_player.stack -= bet_size
                curr_player.insert_round_action(round_name, (action, bet_size))
                self.pot_size += bet_size
                curr_call_size = player_round_bet_size+bet_size
            elif action == 'allin' or action == 'caps':
                aggressor_index = player_index if round(100*(player_round_bet_size+curr_player.stack)) > round(100*curr_call_size) else aggressor_index
                
                curr_player.stack -= curr_player.stack
                curr_player.insert_round_action(round_name, (action, bet_size))
                self.pot_size += curr_player.stack
                curr_call_size = player_round_bet_size+curr_player.stack if round(100*curr_call_size) < round(100*(player_round_bet_size+curr_player.stack)) else curr_call_size
                allin_players.append(player_index)
            else:
                raise ValueError(f"Invalid action {action}: Must be one of {ACTION_NAMES}")
                
            i += 1
        
        print(f"Round {round_name} is over")
        
        if len(self.players) == 1:
            print(f"{self.players[0].name} wins the round")
            return True
        
        return False
    
    def showdown(self):
        # Showdown
        player_hands = {player: evaluate_hand(player.hole_cards, self.community_cards) for player in self.players}
        winning_player = max(player_hands, key=player_hands.get)
        return winning_player
        
    def end_game(self, winning_player: PokerPlayer):
        # End the game
        winning_player.stack += self.pot_size
        self.pot_size = 0
        print(f"{winning_player.name} wins the game")
        
    def play_game(self):
        # Play the entire game
        self.shuffle_deck()
        self.deal_hole_cards()
        for round_name in ROUND_NAMES:
            if self.play_round(round_name):
                break
        if len(self.players) > 1:
            print("Showdown")
            winning_player = self.showdown()
        else:
            winning_player = self.players[0]
        self.end_game(winning_player)

class PokerProbabilityCalculator():
    def __init__(self, hole_cards = [], community_cards = [], num_players = None, num_simulations = 1000):
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.num_players = num_players
        self.num_simulations = num_simulations
        
        self.deck = CARDS.copy()
        try: 
            for card in self.hole_cards + self.community_cards:
                self.deck.remove(card)
        except:
            raise ValueError(f"Invalid cards {self.hole_cards + self.community_cards}\n {self.deck}")
            
    def set(self, hole_cards = None, community_cards = None, num_players = None, num_simulations = None):
        self.hole_cards = self.hole_cards if hole_cards is None else hole_cards
        self.community_cards = self.community_cards if community_cards is None else community_cards
        self.num_players = self.num_players if num_players is None else num_players
        self.num_simulations = self.num_simulations if num_simulations is None else num_simulations
        
        try:
            if community_cards is not None or hole_cards is not None:
                self.deck = CARDS.copy()
                for card in self.hole_cards + self.community_cards:
                    self.deck.remove(card)
        except:
            raise ValueError(f"Invalid cards {self.hole_cards + self.community_cards}\n {self.deck}")
            
    def poker_probability_calculator(self):
        if len(self.hole_cards) != 2:
            raise ValueError("Invalid number of hole cards")
        elif len(self.community_cards) > 5:
            raise ValueError("Invalid number of community cards")
            
        main_player_wins = 0
        other_players_wins = 0
        tie_players_count = []
        
        for i in range(self.num_simulations):
            temp_deck, community_cards = self.deck.copy(), self.community_cards.copy()
            np.random.shuffle(temp_deck)
            other_players_cards = self.get_random_cards_from_deck(temp_deck, self.num_players-1)
            community_cards += [temp_deck.pop() for _ in range(5-len(self.community_cards))]
            
            main_player_hand_value = evaluate_hand(self.hole_cards, community_cards)
            other_players_hand_values = [evaluate_hand(cards, community_cards) for cards in other_players_cards]
            
            if main_player_hand_value > max(other_players_hand_values):
                main_player_wins += 1
            elif main_player_hand_value < max(other_players_hand_values):
                other_players_wins += 1
            elif main_player_hand_value == max(other_players_hand_values):
                tie_players_count.append(other_players_hand_values.count(main_player_hand_value)+1)
            else:
                print("Should not run")
                raise ValueError("Should not run")
            
        avg_tied_players = sum(tie_players_count)/len(tie_players_count) if tie_players_count else 1
        win_rate = main_player_wins/self.num_simulations
        lose_rate = other_players_wins/self.num_simulations
        tie_rate = len(tie_players_count)/self.num_simulations
        
        return win_rate, lose_rate, tie_rate, avg_tied_players
            
    def get_random_cards_from_deck(self, deck, num_players):
        return [[deck.pop(), deck.pop()] for _ in range(num_players)]
            
    
                            
            
def test_action_function(input_data):
    input_string = input("Enter action in format (ACTIONNAME_BETSIZE): ")
    action = input_string.split("_")
    action_name = action[0]
    bet_size = float(action[1])
    if action_name not in ACTION_NAMES:
        raise ValueError("Invalid action name")
    elif len(action) != 2:
        raise ValueError("Invalid action format")
    elif bet_size < 0:
        raise ValueError("Invalid bet size")
    return action_name, bet_size
    
            
if __name__ == "__main__":
    poker_calc = PokerProbabilityCalculator(["2h", "3h"], [], 6, 10000)
    print(poker_calc.poker_probability_calculator())
    print(poker_calc.poker_probability_calculator())
    print(poker_calc.poker_probability_calculator())
    print(poker_calc.poker_probability_calculator())
    
    
    # player1 = PokerPlayer("Alice", 1, 1000, action_func=test_action_function)
    # player2 = PokerPlayer("Bob", 2, 1000, action_func=test_action_function)
    # player3 = PokerPlayer("Cindy", 3, 1000, action_func=test_action_function)
    # player4 = PokerPlayer("David", 4, 1000, action_func=test_action_function)
    
    # poker_game = TexasHoldemPokerGame([player1, player2, player3, player4], 5, 10)
    # poker_game.shuffle_deck()
    # poker_game.deal_hole_cards()
    
    # poker_game.play_round("PREFLOP")
    # poker_game.play_round("FLOP")
    # poker_game.play_round("TURN")
    # poker_game.play_round("RIVER")
    

        