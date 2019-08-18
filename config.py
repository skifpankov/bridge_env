import numpy as np

NUM_PLAYERS: int = 4
NUM_PAIRS: int = 2
NUM_CARDS: int = 52

AUCTION_SPACE_SIZE: int = 38
AUCTION_HISTORY_SIZE: int = 322
NUM_PLAYS = int(NUM_CARDS / NUM_PLAYERS)

PASS_IDX = 35
DOUBLE_IDX = 36
REDOUBLE_IDX = 37
REDOUBLE_RANGE = [DOUBLE_IDX, REDOUBLE_IDX]

Seat = list(range(NUM_PLAYERS))
Suit = list(range(4))
Strain = list(range(5))
Rank = list(range(2, 15))
FULL_DECK = list(range(NUM_CARDS))

Seat2Group = {0: 0, 1: 1, 2: 0, 3: 1} # or s%2

Seat2str = {0: "N", 1: "E", 2: "S", 3: "W"}

# Suit2str is defined in ascending order of Trumps, i.e. Clubs, Diamonds, Hears, Spades (C,D,H,S)
Suit2str = {0: "C", 1: "D", 2: "H", 3: "S"}

# Strain2str is created by merging Suit2str with an extra element for "No Trump" (N)
Strain2str = {**Suit2str, **{4: "N"}}

Rank2str = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}

# Convenient for parallel processing
# ScoreScale = {0: 20, 1: 20, 2: 30, 3: 30, 4: 30}
ScoreScale = np.array([20, 20, 30, 30, 30])
# ScoreBias = {0: 0, 1: 0, 2: 0, 3: 0, 4: 10}
ScoreBias = np.array([0, 0, 0, 0, 10])

MAXNOOFBOARDS = 200