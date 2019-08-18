from typing import Union
import numpy as np

from config import NUM_PAIRS, NUM_PLAYERS, Seat2Group, Strain2str


class Contract(object):
    """
    A "utility" class that stores the attributes of the contract (the winning bid),
    and provides methods for setting and extracting this information
    """

    def __init__(self):

        self._declarer = None
        self._declarer_side = None
        self._suit = None
        self._level = None
        self._double = False
        self._redouble = False
        self._pair_vulnerability = None
        self._player_vulnerability = None

        self.reset()

    def reset(self) -> None:
        """
        Resets all the attributes of the class instance

        :return: None
        """

        self._declarer = None
        self._declarer_side = None
        self._suit = None
        self._level = None
        self._double = False
        self._redouble = False
        self._pair_vulnerability = np.zeros(NUM_PAIRS, dtype=np.uint8)
        self._player_vulnerability = np.zeros(NUM_PLAYERS, dtype=np.uint8)

    @property
    def declarer(self):
        return self._declarer

    @declarer.setter
    def declarer(self, declarer: int):

        # setting the declarer
        self._declarer = declarer

        # setting the declarer's side
        self.declarer_side = Seat2Group[declarer]

    @property
    def declarer_side(self):
        return self._declarer_side

    @declarer_side.setter
    def declarer_side(self, declarer_side: int):

        # assigning the value of declarer's side
        self._declarer_side = declarer_side

    @property
    def suit(self):
        return self._suit

    @property
    def suit_as_str(self) -> Union[str, None]:
        """
        Returns the suit of the contract as string (if suit is set), or None (if suit is not set)

        :return: either a string representation of the contract's suit, or None
        """

        if self.suit is not None:
            return Strain2str[self.suit]
        else:
            return None

    @suit.setter
    def suit(self, suit: int):

        # assigning the value of the contract's suit (which also includes "No Trump")
        self._suit = suit

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level: int):

        # assigning the value of the contract's level (the number of tricks above 6)
        self._level = level

    @property
    def double(self):
        return self._double

    @double.setter
    def double(self, double: bool):

        # specifying whether the contract was doubled
        self._double = double

    @property
    def redouble(self):
        return self._redouble

    @redouble.setter
    def redouble(self, redouble):

        # we can only redouble if double is True; if not, redouble should be set to False
        if self.double:
            self.redouble = redouble
        else:
            self.redouble = False

    @property
    def vulnerability(self):
        return self._pair_vulnerability

    @vulnerability.setter
    def vulnerability(self, vulnerability: np.array):

        # setting the value of vulnerability
        self._pair_vulnerability = vulnerability

    def from_bid(self, bid: int, double: bool = False, redouble: bool = False) -> None:
        """
        Assigns the contract's trump, level, double and redouble from the winning bid

        :param bid: an int between 0 and 34

        :return: None
        """

        # bids go up in the following order: [1C, 1D, 1H, 1S, 1N, ..., 7N] (i.e.
        # they are in ascending order)

        # suit is the remainder of dividing the bid by 5
        self.suit = bid % 5
        self.level = bid // 5 + 1

        self.double = double
        self.redouble = redouble
