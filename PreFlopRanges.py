import math
import CardsEvaluation as ce
import random
import matplotlib.pyplot as plt

class Player():
    def __init__(self, playerRange: dict = None):
        self.cards = RemoveRandomCardsWithRange(playerRange)
        self.playerRange = playerRange
        return

class PokerPlayer():
    def __init__(self, name, stackSize, cards, betSize = 0, actionFunc = None):
        self.finishedTurn = False
        self.name = name
        self.stackSize = stackSize
        self.cards = cards
        self.betSize = betSize
        if actionFunc == None:
            self.actionFunc = lambda: input('Player Action: ')
            self.isBot = False
        else:
            self.actionFunc = actionFunc
            self.isBot = True
        return
    
    def __str__(self):
        return f'Name: {self.name}\nStack Size: {self.stackSize}\nCards: {self.cards}\nBet Size: {self.betSize}'
    
    def __eq__(self, other):
        return self.name == other.name
    
class Table():
    def __init__(self, numPlayers, playerStack=1000, potSize = 0):
        self.numPlayers = numPlayers
        self.players = [PokerPlayer('Player' + str(i), playerStack, RemoveRandomCards(2)) for i in range(numPlayers)]
        self.communityCards = []
        self.potSize = potSize
        return
    
    def PlaceCommunityCards(self, numCards):
        self.communityCards += RemoveRandomCards(numCards)
        
    def ShowTable(self):
        for player in self.players:
            print(player)
            print('-------------------')
        print(f'Community Cards: {self.communityCards}')
        print(f'Pot Size: {self.potSize}')
        print('-------------------')
        print('-------------------\n')
    
    def RemovePlayer(self, player):
        self.players.remove(player)
        self.numPlayers = len(self.players)
        return
    
    def AddPlayer(self, player):
        self.players.append(player)
        self.numPlayers = len(self.players)
        return
    
class PokerGame():
    def __init__(self, numPlayers, numBots, playerStack, smallBlind, bigBlind):
        self.table = Table(numPlayers, playerStack)
        for i in range(numBots):
            self.table.AddPlayer(PokerPlayer('Bot' + str(i), playerStack, RemoveRandomCards(2), 0, Bot))
        self.smallBlind = smallBlind
        self.bigBlind = bigBlind
        self.smallBlindIndex = 0
        return
    
    def ShowStatus(self):
        self.table.ShowTable()
        return
    
    def ResetPlayerFinishing(self, playerException):
        for player in self.table.players:
            player.finishedTurn = False if playerException != player else True
    def ResetPlayerBets(self):
        for player in self.table.players:
            player.betSize = 0
        return
    
    def GetResults(self):
        maxHandValue = 0
        winnerPlayer = None
        for player in self.table.players:
            handValue = HandDetection(player.cards + self.table.communityCards)
            if handValue > maxHandValue:
                maxHandValue = handValue
                winnerPlayer = player
        
        print('WINNER: ', winnerPlayer.name)
        exit()
        
        return winnerPlayer
        
# Bot(myCards, communityCards, numPlayers, potSize, stackSize, callSize, minRaise, accuracy=10000):
    def Round(self, init = False):
        i = 0
        roundRaise = 0
        roundFinished = False
        
        while not roundFinished:
            index = GetCircularIndex(self.smallBlindIndex, i, self.table.numPlayers)
            player = self.table.players[index]
            
            callSize = roundRaise - player.betSize
            
            if self.table.communityCards == [] and init and (index == self.smallBlindIndex or index == GetCircularIndex(self.smallBlindIndex, 1, self.table.numPlayers)):
                if index == self.smallBlindIndex:
                    action = 'RAISE_BY' + str(self.smallBlind)
                elif index == GetCircularIndex(self.smallBlindIndex, 1, self.table.numPlayers):
                    action = 'RAISE_BY' + str(self.bigBlind)
                    init = False
                
            elif player.isBot:
                action = player.actionFunc(player.cards, self.table.communityCards, self.table.numPlayers, self.table.potSize, player.stackSize, callSize, 2*roundRaise)
            else:
                action = player.actionFunc()
                
            if action == 'FOLD':
                self.table.RemovePlayer(player)
                i -= 1
                
            elif action == 'CALL':
                self.table.potSize += callSize
                player.stackSize -= callSize
                player.betSize = roundRaise
                player.finishedTurn = True
                
            elif action == 'ALL-IN':
                self.table.potSize += player.stackSize
                if roundRaise < player.stackSize:
                    roundRaise = player.stackSize
                player.stackSize = 0
                player.betSize = player.stackSize
                self.ResetPlayerFinishing(player)
                
            elif action[:5] == 'RAISE':
                raiseBy = float(action[8:])
                self.table.potSize += raiseBy
                player.stackSize -= raiseBy
                roundRaise = raiseBy + player.betSize
                player.betSize += raiseBy
                self.ResetPlayerFinishing(player)
                
            roundFinished = all([player.finishedTurn for player in self.table.players])
            
            i += 1
            self.ShowStatus()
            print(f'Player: {player.name}')
            print('ACTION: ', action)
            print('-------------------\n\n')
        
        print('Round Over')
        if self.table.numPlayers == 1:
            print('GAME OVER: WINNER: ', self.table.players[0].name)
            exit()
        self.ResetPlayerBets()
        return None
    
    
    
    
    
    
SUITS = ['s', 'h', 'c', 'd']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

DECK = [rank + suit for rank in RANKS for suit in SUITS]

def GetCircularIndex(start, index, length):
    return (length+start+index)%length

def ResetDeck(deck = None):   
    global DECK
    if deck == None:
        deck = [rank + suit for rank in RANKS for suit in SUITS]
        return deck
    DECK = [rank + suit for rank in RANKS for suit in SUITS]
    return None

def RemoveCards(cards: list[str], deck = None):
    global DECK
    if deck == None:
        for card in cards:
            deck.remove(card)
        return cards, deck
    
    for card in cards:
        DECK.remove(card)
    return cards

def RemoveRandomCardsWithRange(playerRange: dict, deck = None):
    holeCards = list(playerRange.keys())
    prob = list(playerRange.values())
    holdCard = [None, None]
    card1NotInDeck = True
    card2NotInDeck = True
    while card1NotInDeck or card2NotInDeck:
        holdCard = random.choices(holeCards, weights=prob)[0]
        card1NotInDeck, card2NotInDeck = holdCard[0] not in DECK, holdCard[1] not in DECK
    return RemoveCards(list(holdCard))

def RemoveRandomCards(numCards):
    global DECK
    randints = random.sample(range(len(DECK)), numCards)
    return [DECK.pop(i) for i in sorted(randints, reverse=True)]

def Combination(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

def GetPlayerRange():
    playerRange = {}
    totalPossibleHands = Combination(len(DECK), 2)
    for i in range(len(DECK)-1):
        for j in range(i+1, len(DECK)):
            playerRange[(DECK[i], DECK[j])] = 1/totalPossibleHands
    return playerRange

def ConvertRank(rank):
    if rank == 'T':
        return 10
    elif rank == 'J':
        return 11
    elif rank == 'Q':
        return 12
    elif rank == 'K':
        return 13
    elif rank == 'A':
        return 1
    else:
        return int(rank)

def HandDetection(cards: list[str]) -> int:
    if len(cards) != 7:
        raise ValueError('HandDetection: Invalid number of cards\n Got {} instead of 7'.format(len(cards)))
    avail = [ce.Card(ConvertRank(card[0]), card[1].upper()) for card in cards]
    #print(avail)
    return ce.HandEvaluation(avail)

def HandDistribution(iterations: int):
    data = []
    for i in range(iterations):
        cards = random.sample(DECK, 7)
        value = HandDetection(cards)
        data.append(value)
    
    plt.hist(data, bins=1000)
    plt.xlabel("Hand Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Hand Values")
    plt.show()
    
    return

# Output win probability of player with given cards against opponent with given cards
def SimulationalProbability(myCards, communityCards, numPlayers, iterations):
    if len(communityCards) > 5:
        raise ValueError('SimulationalProbability: Invalid number of community cards\n Got {} instead of 5'.format(len(communityCards)))
    DECK_COPY = DECK.copy()
    
    playerRange = GetPlayerRange()
    tiedPlayer = 0
    win = 0
    loss = 0
    tie = 0
    cards = []
    for i in range(iterations):
        players = [Player(playerRange) for i in range(numPlayers-1)]
        
        if len(communityCards) < 5:
            cards, DECK_COPY = RemoveRandomCards(5 - len(communityCards), deck = DECK_COPY)
            cards = cards + communityCards
        elif len(communityCards) == 5:
            cards = communityCards
            
            
        myValue = HandDetection(myCards + cards)
        oppValues = [HandDetection(player.cards + cards) for player in players]
        strongestOppValue = max(oppValues)
        if myValue > strongestOppValue:
            win += 1
        elif myValue < strongestOppValue:
            loss += 1
        else:
            tiedPlayer += oppValues.count(strongestOppValue)+1
            tie += 1
        
        DECK_COPY = ResetDeck(deck = DECK_COPY)
        _, DECK_COPY = RemoveCards(myCards+communityCards, deck = DECK_COPY)
        
    if tie > 0:
        avgTiedPlayers = tiedPlayer/tie
    else:
        avgTiedPlayers = None
        
    return win/iterations, loss/iterations, tie/iterations, avgTiedPlayers

def BetSizeFunction(winProb, lossProb, tieProb, avgTiedPlayers, potSize, profit, callSize):
    avgTiedCostProb = tieProb/avgTiedPlayers
    denom = (winProb - lossProb + avgTiedCostProb)
    if denom == 0:
        denom = 0.0001
    elif denom < 0:
        return None, None
    betSize = (profit + (callSize - potSize)*(winProb + avgTiedCostProb))/denom
    betSizeDer = 1/denom
    return betSize, betSizeDer

def ProfitFunction(winProb, lossProb, tieProb, avgTiedPlayers, potSize, betSize, callSize):
    avgTiedCostProb = tieProb/avgTiedPlayers
    profit = (potSize-callSize)*(winProb+avgTiedCostProb) - betSize*(winProb + avgTiedCostProb - lossProb)
    return profit

def normalize(betSizeDer):
    return math.exp(-(betSizeDer-1))

def ProbToBet(winProb, lossProb, tieProb, avgTiedPlayers, potSize, stackSize, callSize, minRaise):
    betSize, betSizeDer = BetSizeFunction(winProb, lossProb, tieProb, avgTiedPlayers, potSize, 0, callSize)
    
    if betSize == None and callSize == 0:
        return 0
    if betSize == None:
        return None
    if betSize > stackSize:
        return None
    
    betSize = round(normalize(betSizeDer)*stackSize)
    print(f'betSize: {betSize}, betSizeDer: {betSizeDer}, normalized: {normalize(betSizeDer)}')    
    
    if betSize < minRaise and betSize > callSize:
        minRaiseProfit = ProfitFunction(winProb, lossProb, tieProb, avgTiedPlayers, potSize, minRaise, callSize)
        callSizeProfit = ProfitFunction(winProb, lossProb, tieProb, avgTiedPlayers, potSize, callSize, callSize)
        if minRaiseProfit > callSizeProfit:
            betSize = minRaise
        elif callSizeProfit >= minRaiseProfit:
            betSize = callSize
    elif betSize < callSize:
        callSizeProfit = ProfitFunction(winProb, lossProb, tieProb, avgTiedPlayers, potSize, callSize, callSize)
        if callSizeProfit > 0:
            betSize = callSize
        else:
            return None
    
    return betSize

def Bot(myCards, communityCards, numPlayers, potSize, stackSize, callSize, minRaise, accuracy=10000):
    if stackSize == 0:
        return 'ALR_ALL_IN'
    winProb, lossProb, tieProb, avgTiedPlayers = SimulationalProbability(myCards, communityCards, numPlayers, accuracy)
    print(f'winProb: {winProb}, lossProb: {lossProb}, tieProb: {tieProb}')
    if avgTiedPlayers == None:
        avgTiedPlayers = 1
        tieProb = 0
    betSize = ProbToBet(winProb, lossProb, tieProb, avgTiedPlayers, potSize, stackSize, callSize, minRaise)
    
    if betSize == None:
        return 'FOLD'
    
    if math.isclose(betSize, callSize):
        return 'CALL'
    
    if math.isclose(betSize, minRaise) or betSize > minRaise:
        return f'RAISE_BY{betSize}'
    
    if math.isclose(betSize, stackSize, rel_tol=0.05):
        return 'ALL-IN'

if __name__ == '__main__':
    numPlayers = 1
    numBots = 1
    playerStack = 1000
    smallBlind = 10
    bigBlind = 20
    
    game = PokerGame(numPlayers, numBots, playerStack, smallBlind, bigBlind)
    game.Round(init=True)
    game.table.PlaceCommunityCards(3)
    game.Round()
    game.table.PlaceCommunityCards(1)
    game.Round()
    game.table.PlaceCommunityCards(1)
    game.Round()
    game.GetResults()
    
    
    
    # playerRange = GetPlayerRange()
    
    # # Random 7 cards from DECK
    # numPlayers = 6
 
    # myCards = RemoveCards(['As', 'Ah'])
    # communityCards = RemoveCards(['Ks', 'Qs', 'Js'])

    # print(SimulationalProbability(myCards, communityCards, numPlayers, 10000))
    
    # numPlayers = 2
    # bigBlind = 20
    # smallBlind = 10
    # potSize = 30
    # roundRaise = bigBlind
    # myStackSize = 1000 - bigBlind
    # botStackSize = 1000 - smallBlind
    # callSize = bigBlind - smallBlind
    # minRaise = bigBlind*2
    # myCards = RemoveRandomCards(2)
    # botCards = RemoveRandomCards(2)
    # communityCards = RemoveRandomCards(3)
    
    # PreFlop
    # while True:
    #     print(f'My Cards: {myCards}')
    #     print(f'My Stack Size: {myStackSize}')
    #     print(f'Bot Cards: {botCards}')
    #     print(f'Bot Stack Size: {botStackSize}')
    #     print(f'Community Cards: {communityCards}')
    #     print(f'Pot Size: {potSize}')
    #     action = Bot(botCards, communityCards, numPlayers, potSize, botStackSize, callSize, minRaise)
    #     print("Bot: ", action)
        
    #     if action == 'FOLD':
    #         print('Bot Folds: GAME OVER')
    #         exit()
            
    #     elif action == 'CALL':
    #         potSize += callSize
    #         botStackSize -= callSize
    #         break
        
    #     elif action == 'ALL-IN':
    #         potSize += botStackSize
    #         if roundRaise < botStackSize:
    #             roundRaise = botStackSize
    #         botStackSize = 0 
    #         raiseBy = botStackSize
            
    #     elif action[:5] == 'RAISE':
    #         raiseBy = float(action[8:])
    #         potSize += raiseBy
    #         botStackSize -= raiseBy
    #         roundRaise = raiseBy
        
    #     callSize = raiseBy - callSize
        
    #     print(f'Bot Stack Size: {botStackSize}')
    #     print(f'Pot Size: {potSize}')
    #     print(f'minRaise: {2*roundRaise}')
    #     print(f'callSize: {callSize}')
        
    #     action = input('Player Action: ')
        
    #     if action == 'FOLD':
    #         print('Player Folds: GAME OVER')
    #         exit()
            
    #     elif action == 'CALL':
    #         potSize += callSize
    #         myStackSize -= callSize
    #         break
        
    #     elif action == 'ALL-IN':
    #         potSize += myStackSize
    #         if roundRaise < myStackSize:
    #             roundRaise = myStackSize
    #         myStackSize = 0 
    #         raiseBy = myStackSize
            
    #     elif action[:5] == 'RAISE':
    #         raiseBy = float(action[8:])
    #         potSize += raiseBy
    #         myStackSize -= raiseBy
    #         roundRaise = raiseBy
            
        
    #     callSize = roundRaise
    
    # print('Round 1 Over')
    # print("--------------------")
    # roundRaise = 0
    # callSize = 0
    # communityCards += RemoveRandomCards(1)
