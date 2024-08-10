
DEBUG = False

LOWEST_HIGH_CARD = 0
LOWEST_PAIR = 9072
LOWEST_TWO_PAIR = 17405
LOWEST_THREE_OF_A_KIND = 19128
LOWEST_STRAIGHT = 20417
LOWEST_FLUSH = 20427
LOWEST_FULL_HOUSE = 67083
LOWEST_FOUR_OF_A_KIND = 67251
LOWEST_STRAIGHT_FLUSH = 67406
# MAX STRAIGHT FLUSH = 67415

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    
    def __repr__(self):
        return str(self.rank) + self.suit
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    def __lt__(self, other):
        return self.rank < other.rank
    def __gt__(self, other):
        return self.rank > other.rank
    
    def __add__(self, value):
        return Card(self.rank + value, self.suit)


# def GetStraightCards(sortedCards: list[Card]) -> list[Card]:
#     # Must be sorted
#     straightRanks = [sortedCards[0]]
#     for i in range(1, len(sortedCards)):
#         if sortedCards[i] == sortedCards[i-1] + 1:
#             straightRanks.append(sortedCards[i])
#         elif len(straightRanks) < 5:
#             straightRanks = []
#         if i == len(sortedCards) - 1:
#             if sortedCards[i] == 13 and sortedCards[0] == 1:
#                 straightRanks.append(1)
#                 straightRanks = sorted(straightRanks)
#             if len(straightRanks) < 5:
#                 straightRanks = []
    
#     return straightRanks

def HandEvaluation(cards: list[Card]):
    if len(cards) != 7:
        raise ValueError(f'HandEvaluation: Invalid number of cards\n Got {len(cards)} instead of 7')
    
    sortedCards = sorted(cards)
    
    rankInfo = {}
    suitInfo = {}
    for i in range(len(sortedCards)):
        if sortedCards[i].rank not in rankInfo:
            rankInfo[sortedCards[i].rank] = [i]
        else:
            rankInfo[sortedCards[i].rank].append(i)
        
        if sortedCards[i].suit not in suitInfo:
            suitInfo[sortedCards[i].suit] = [i]
        else:
            suitInfo[sortedCards[i].suit].append(i)
    
    sortedRanks = sorted(rankInfo.keys())
    
    # Pair, Two Pair, Three of a Kind, Four of a Kind Analysis
    pair1 = None
    pair2 = None
    threeKind = None
    fourKind = None
    for rank in sortedRanks:
        if len(rankInfo[rank]) == 4:
            fourKind = [sortedCards[i] for i in rankInfo[rank]]
        elif len(rankInfo[rank]) == 3:
            if not threeKind:
                threeKind = [sortedCards[i] for i in rankInfo[rank]]
            else:
                pair1 = threeKind
                threeKind = [sortedCards[i] for i in rankInfo[rank]]
        elif len(rankInfo[rank]) == 2:
            if not pair1:
                pair1 = [sortedCards[i] for i in rankInfo[rank]]
            else:
                pair2 = pair1
                pair1 = [sortedCards[i] for i in rankInfo[rank]]
    
    if 1 in rankInfo:
        if len(rankInfo[1]) == 2 and pair1 and pair2:
            pair2 = pair1
            pair1 = [sortedCards[i] for i in rankInfo[1]]
        elif len(rankInfo[1]) == 3:
            if pair1 and len(pair1)==3:
                threeKind, pair1 = pair1, threeKind
                
                
            else:
                threeKind = [sortedCards[i] for i in rankInfo[1]]
            
        elif len(rankInfo[1]) == 4:
            fourKind = [sortedCards[i] for i in rankInfo[1]]
    # print(rankInfo)
    # print(threeKind, pair1, pair2, fourKind)
                
    # Straight Analysis
    sortedRanks = sorted(rankInfo.keys())
    straightRanks = []
    for i in range(1, len(sortedRanks)):
        if sortedRanks[i] == sortedRanks[i-1] + 1:
            if len(straightRanks) == 0:
                straightRanks.append(sortedRanks[i-1])
            straightRanks.append(sortedRanks[i])
        elif len(straightRanks) < 5:
            straightRanks = []
        elif len(straightRanks) >= 5:
            break
        if i == len(sortedRanks) - 1:
            if sortedRanks[i] == 13 and sortedRanks[0] == 1 and len(straightRanks) >= 4 and straightRanks[-4] == 10:
                straightRanks.append(1)
            if len(straightRanks) < 5:
                straightRanks = []
    
    # Flush Analysis
    flushSuit = None
    for suit in suitInfo:
        if len(suitInfo[suit]) >= 5:
            flushSuit = suit
    
    # Straight Flush Analysis
    straightFlush = []
    if flushSuit:
        flushCards = [sortedCards[i] for i in suitInfo[flushSuit]]
        straightFlush = []
        for i in range(1, len(flushCards)):
            if flushCards[i] == flushCards[i-1] + 1:
                if len(straightFlush) == 0:
                    straightFlush.append(flushCards[i-1])
                straightFlush.append(flushCards[i])
            elif len(straightFlush) < 5:
                straightFlush = []
            if i == len(flushCards) - 1:
                if flushCards[i].rank == 13 and flushCards[0].rank == 1 and len(straightFlush) >= 4 and straightFlush[-4].rank == 10:
                    straightFlush.append(flushCards[0])
                if len(straightFlush) < 5:
                    straightFlush = []
    
    # Check if Straight Flush
    if len(straightFlush):
        # return 'Straight Flush'
        return StraightFlushValue(straightFlush)
    
    # Check if Four of a Kind
    if fourKind:
        # return 'Four of A Kind'
        return FourOfAKindValue(fourKind, sortedCards)
        
    # Check if Full House
    if pair1 and threeKind:
        # return 'Full House'
        return FullHouseValue(threeKind, pair1[:2])
    
    # Check if Flush
    if flushSuit:
        # return 'Flush'
        return FlushValue(flushCards)
    
    # Check if Straight
    if len(straightRanks):
        # return 'Straight'
        return StraightValue(straightRanks)
    
    # Check if Three of a Kind
    if threeKind:
        # return 'Three of A Kind'
        return ThreeOfAKindValue(threeKind, sortedCards)
        
    # Check if Two Pair
    if pair1 and pair2:
        # return 'Two Pair'
        return TwoPairValue(pair1, pair2, sortedCards)
    
    # Check if Pair
    if pair1:
        # return 'Pair'
        return PairValue(pair1, sortedCards)
    
    # Check if High Card
    # return 'High Card'
    return HighcardValue(sortedCards)

def StraightFlushValue(straightFlush):
    if len(straightFlush) < 5:
        raise ValueError(f'StraightFlushValue: Invalid number of cards\n Got {len(straightFlush)} instead of 5')
    
    if straightFlush[-1].rank == 1:
        straightFlush[-1] = straightFlush[-1]+13
    
    if DEBUG:
        print(straightFlush)
    
    return LOWEST_STRAIGHT_FLUSH + (straightFlush[-1].rank-5)

def FourOfAKindValue(fourKind, sortedCards):
    if len(sortedCards) != 7:
        raise ValueError(f'FourOfAKindValue: Invalid number of cards\n Got {len(sortedCards)} instead of 7')
    
    sortedCards = [card for card in sortedCards if card not in fourKind]
    
    if fourKind[0].rank == 1:
        fourKind = [fourKind[0]+13, fourKind[1]+13, fourKind[2]+13, fourKind[3]+13]
    
    elif sortedCards[0].rank == 1:
        sortedCards.append(sortedCards[0]+13)
        sortedCards.pop(0)
    
    hand = sortedCards[-1:] + fourKind
    if DEBUG:
        print(hand)
    
    return LOWEST_FOUR_OF_A_KIND + (fourKind[0].rank-2)*12 + (sortedCards[-1].rank-3)

def FullHouseValue(threeKind, pair):
    if threeKind[0].rank == 1:
        threeKind = [threeKind[0]+13, threeKind[1]+13, threeKind[2]+13]
    elif pair[0].rank == 1:
        pair = [pair[0]+13, pair[1]+13]
        
    hand = pair + threeKind
    if DEBUG:
        print(hand)
    
    return LOWEST_FULL_HOUSE + (threeKind[0].rank-2)*13 + (pair[0].rank-2)
    

def FlushValue(flushCards):
    if len(flushCards) < 5:
        raise ValueError(f'FlushValue: Invalid number of cards\n Got {len(flushCards)} instead of 5')
    
    if flushCards[0].rank == 1:
        flushCards.append(flushCards[0]+13)
        flushCards.pop(0)
    
    hand = flushCards[-5:]
    if DEBUG:
        print(hand)
    
    return LOWEST_FLUSH + (flushCards[-1].rank-7)*(9**3)*8 + (flushCards[-2].rank-5)*(9**2)*8 + (flushCards[-3].rank-4)*9*8 + (flushCards[-4].rank-3)*8 + (flushCards[-5].rank-2)
    

def StraightValue(straightRanks):
    if len(straightRanks) < 5:
        raise ValueError(f'StraightValue: Invalid number of cards\n Got {len(straightRanks)} instead of 5')
    
    if straightRanks[-1] == 1:
        straightRanks[-1] = 14
    
    if DEBUG:
        print(straightRanks)
    
    return LOWEST_STRAIGHT + (straightRanks[-1] - 5)

def ThreeOfAKindValue(threeKind, sortedCards):
    if len(sortedCards) != 7:
        raise ValueError(f'ThreeOfAKindValue: Invalid number of cards\n Got {len(sortedCards)} instead of 7')
    
    sortedCards = [card for card in sortedCards if card not in threeKind]
    
    if threeKind[0].rank == 1:
        threeKind = [threeKind[0]+13, threeKind[1]+13, threeKind[2]+13]
    
    elif sortedCards[0].rank == 1:
        sortedCards.append(sortedCards[0]+13)
        sortedCards.pop(0)
        
    hand = sortedCards[-2:] + threeKind
    if DEBUG:
        print(hand)
    
    return LOWEST_THREE_OF_A_KIND + (threeKind[0].rank-2)*10*10 + (sortedCards[-1].rank-5)*10 + (sortedCards[-2].rank-4)

def TwoPairValue(pair1, pair2, sortedCards):
    if len(sortedCards) != 7:
        raise ValueError(f'TwoPairValue: Invalid number of cards\n Got {len(sortedCards)} instead of 7')
    
    sortedCards = [card for card in sortedCards if card not in pair1 and card not in pair2]
    
    if pair1[0].rank == 1:
        pair1 = [pair1[0]+13, pair1[1]+13]
    
    elif sortedCards[0].rank == 1:
        sortedCards.append(sortedCards[0]+13)
        sortedCards.pop(0)
        
    hand = sortedCards[-1:] + pair1 + pair2
    if DEBUG:
        print(hand)
    
    return LOWEST_TWO_PAIR - 3 + (pair1[0].rank - 3)*12*12 + (pair2[0].rank - 2)*12 + (sortedCards[-1].rank - 3)


def PairValue(pair, sortedCards):
    if len(sortedCards) != 7:
        raise ValueError(f'PairValue: Invalid number of cards\n Got {len(sortedCards)} instead of 7')
    
    sortedCards = [card for card in sortedCards if card not in pair]
    
    if pair[0].rank == 1:
        pair = [pair[0]+13, pair[1]+13]
        sortedCards = sortedCards[2:]
    
    elif sortedCards[0].rank == 1:
        sortedCards.append(sortedCards[0]+13)
        sortedCards.pop(0)
        
    hand = sortedCards[-3:] + pair
    if DEBUG: 
        print(hand)
    
    return LOWEST_PAIR + (pair[0].rank - 2)*8*9*9 + (sortedCards[-1].rank - 7)*9*9 + (sortedCards[-2].rank - 5)*9 + (sortedCards[-3].rank - 4)
    
    
    

def HighcardValue(highCards):
    if len(highCards) != 7:
        raise ValueError(f'HighcardValue: Invalid number of cards\n Got {len(highCards)} instead of 7')
    
    if highCards[0].rank == 1:
        highCards.append(highCards[0]+13)
        highCards.pop(0)
    
    hand = highCards[2:]
    if DEBUG:
        print(hand)
    
    return LOWEST_HIGH_CARD + (highCards[6].rank-9)*7*(6**3) + (highCards[5].rank-8)*7*(6**2) + (highCards[4].rank-7)*7*6 + (highCards[3].rank-5)*6 + (highCards[2].rank-4)

if __name__ == "__main__":
    # Lowest High Card
    cards = [Card(2, 'S'), Card(3, 'H'), Card(4, 'D'), Card(5, 'S'), Card(7, 'H'), Card(8, 'D'), Card(9, 'H')]
    print("Lowest High Card: \t", HandEvaluation(cards))
    
    # Highest High Card
    cards = [Card(7, 'S'), Card(8, 'H'), Card(9, 'D'), Card(11, 'S'), Card(12, 'H'), Card(13, 'D'), Card(1, 'H')]
    print("Highest High Card: \t", HandEvaluation(cards))
    
    # Lowest Pair
    cards = [Card(2, 'S'), Card(2, 'H'), Card(3, 'D'), Card(4, 'S'), Card(5, 'H'), Card(7, 'D'), Card(8, 'H')]
    print("Lowest Pair: \t\t", HandEvaluation(cards))
    
    # Highest Pair
    cards = [Card(1, 'S'), Card(1, 'H'), Card(13, 'D'), Card(12, 'S'), Card(11, 'H'), Card(9, 'D'), Card(8, 'H')]
    print("Highest Pair: \t\t", HandEvaluation(cards))
    
    # Lowest Two Pair
    cards = [Card(2, 'S'), Card(2, 'H'), Card(3, 'D'), Card(3, 'S'), Card(4, 'H'), Card(5, 'D'), Card(7, 'H')]
    print("Lowest Two Pair:\t", HandEvaluation(cards))
    
    # Highest Two Pair
    cards = [Card(1, 'S'), Card(1, 'H'), Card(13, 'D'), Card(13, 'S'), Card(12, 'H'), Card(12, 'D'), Card(11, 'H')]
    print("Highest Two Pair: \t", HandEvaluation(cards))
    
    # Lowest Three of a Kind
    cards = [Card(2, 'S'), Card(2, 'H'), Card(2, 'D'), Card(3, 'S'), Card(4, 'H'), Card(5, 'D'), Card(7, 'H')]
    print("Lowest Three of a Kind:\t", HandEvaluation(cards))
    
    # Highest Three of a Kind
    cards = [Card(1, 'S'), Card(1, 'H'), Card(1, 'D'), Card(13, 'S'), Card(12, 'H'), Card(11, 'D'), Card(9, 'H')]
    print("Highest Three of a Kind:", HandEvaluation(cards))
    
    # Lowest Straight
    cards = [Card(1, 'S'), Card(2, 'H'), Card(3, 'D'), Card(4, 'S'), Card(5, 'H'), Card(7, 'D'), Card(8, 'H')]
    print("Lowest Straight: \t", HandEvaluation(cards))
    
    # Highest Straight
    cards = [Card(9, 'S'), Card(10, 'H'), Card(11, 'D'), Card(12, 'S'), Card(13, 'H'), Card(1, 'D'), Card(8, 'H')]
    print("Highest Straight: \t", HandEvaluation(cards))
    
    # Lowest Flush
    cards = [Card(2, 'S'), Card(3, 'S'), Card(4, 'S'), Card(5, 'S'), Card(7, 'S'), Card(8, 'S'), Card(9, 'S')]
    print("Lowest Flush: \t\t", HandEvaluation(cards))
    
    # Highest Flush
    cards = [Card(7, 'H'), Card(8, 'H'), Card(9, 'H'), Card(11, 'H'), Card(12, 'H'), Card(13, 'H'), Card(1, 'H')]
    print("Highest Flush: \t\t", HandEvaluation(cards))
    
    # Lowest Full House
    cards = [Card(2, 'S'), Card(2, 'H'), Card(2, 'D'), Card(3, 'S'), Card(3, 'H'), Card(4, 'D'), Card(5, 'H')]
    print("Lowest Full House: \t", HandEvaluation(cards))
    
    # Highest Full House
    cards = [Card(1, 'S'), Card(1, 'H'), Card(1, 'D'), Card(13, 'S'), Card(13, 'H'), Card(13, 'D'), Card(12, 'H')]
    print("Highest Full House:\t", HandEvaluation(cards))
    
    # Lowest Four of a Kind
    cards = [Card(2, 'S'), Card(2, 'H'), Card(2, 'D'), Card(2, 'C'), Card(3, 'S'), Card(3, 'H'), Card(3, 'D')]
    print("Lowest Four of a Kind: \t", HandEvaluation(cards))
    
    # Highest Four of a Kind
    cards = [Card(1, 'S'), Card(1, 'H'), Card(1, 'D'), Card(1, 'C'), Card(13, 'S'), Card(13, 'H'), Card(13, 'D')]
    print("Highest Four of a Kind:\t", HandEvaluation(cards))
    
    # Lowest Straight Flush
    cards = [Card(1, 'S'), Card(2, 'S'), Card(3, 'S'), Card(4, 'S'), Card(5, 'S'), Card(2, 'D'), Card(2, 'C')]
    print("Lowest Straight Flush: \t", HandEvaluation(cards))
    
    # Highest Straight Flush
    cards = [Card(9, 'H'), Card(10, 'H'), Card(11, 'H'), Card(12, 'H'), Card(13, 'H'), Card(1, 'H'), Card(8, 'H')]
    print("Highest Straight Flush:\t", HandEvaluation(cards))

    
# 8 = Straight Flush   < 67311
# 7 = Four of a Kind   < 67303
# 6 = Full House       < 67148
# 5 = Flush            < 66981
# 4 = Straight         < 20325
# 3 = Three of a Kind  < 20315
# 2 = Two Pair         < 19037
# 1 = Pair             < 17314
# 0 = Highcard         < 9072