from typing import List, Union
import random
import numpy as np
import pandas as pd

from deal import Deal
from copy import deepcopy
from config import *
from bridge_utils import *
from score import precompute_scores_v2


# CHEAT-SHEET FOR BIDS
# 0-34: contract bids [1C, 1D, 1H, 1S, 1NT, ..., 7S, 7NT]
#   35: pass
#   36: double
#   37: redouble


class BridgeEnv(object):
    """
    This class is intended to replicate bidding and playing in bridge
    """

    def __init__(self,
                 bidding_seats=Seat,
                 nmc=20,
                 debug=False,
                 score_mode="IMP"):

        # pre-calculating scores
        self._score_table = precompute_scores_v2(full_version=True)

        # deal is the state
        self.deal = None
        self.one_hot_deal = None
        self.cards = deepcopy(FULL_DECK)
        self.bidding_history = np.zeros(36, dtype=np.uint8) # pass is included in the history
        self.auction_history = np.zeros(AUCTION_HISTORY_SIZE, dtype=np.uint8)
        self.vulnerability = np.zeros(NUM_PAIRS, dtype=np.uint8)

        # bidding elimination signals - i.e. which bids are currently not permitted
        self.elimination_signal = np.zeros(AUCTION_SPACE_SIZE, dtype=np.uint8)

        # playing elimination signals - i.e. which cards each player does not possess
        self.elimination_signal_play = None

        self.n_pass = 0
        self.n_double = 0
        self.n_redouble = 0

        self.nmc = nmc # MC times
        self.max_bid = -1
        self.contract = None
        self.done_bidding = False
        self.done_playing = False
        self.debug = debug
        self.score_mode = score_mode
        self.strain_declarer = {0: {}, 1: {}}
        self.group_declarer = -1

        # checking that all the bidding seats submitted are valid
        self.bidding_seats = sorted(list(set(bidding_seats)))
        for seat in self.bidding_seats:
            if seat not in Seat:
                raise Exception(f'seat {seat} is illegal. bidding_seats argument can only contain '
                                f'values in {Seat}')

        # the index of the first bidder; start from the smallest one by default
        self.turn = self.bidding_seats[0]

        # the index of the first player
        # TODO[ス: finish writing the logic for self.turn_play
        self.turn_play = self.bidding_seats[0]

        # vector of tricks for all players
        self.tricks = np.zeros(NUM_PLAYERS)

        # vector of current scores
        self.score_play = np.zeros(NUM_PLAYERS)

        # TODO[ス: check whether resetting the environment on initialisation can break anything
        # resetting the environment upon initialisation
        # self.reset()

    def set_nmc(self, n):
        self.nmc = n

    def set_mode(self, debug):
        self.debug = debug

    def reset(self, predeal_seats=None, reshuffle=True):  # North and South
        """
        This method resets the environment - namely:
           - clears bidding history
           - generates new vulnerabilities
           - resets elimination signals (i.e. indicator of actions which cannot be performed)
           -

        :param predeal_seats: if not None, allocate cards to those seats. e.g. [0, 1] stands for
        North and East
        :param reshuffle: whether reshuffle the hands for the predeal seats
        :return: deal
        """

        # resetting bidding history
        # 1C 1D 1H 1S 1N ... 7N (PASS - not considered)
        self.bidding_history = np.zeros(36, dtype=np.uint8)

        # generating vulnerabilities
        self.vulnerability = (np.random.rand(NUM_PAIRS) > 0.5).astype(int)

        # resetting auction_history
        self.auction_history = 0 * self.auction_history

        # resetting elimination_history - doubles and redoubles are not allowed at the start
        self.elimination_signal[REDOUBLE_RANGE] = 1

        # resetting various counts
        self.max_bid = -1
        self.contract = None
        self.n_pass = 0
        self.n_double = 0
        self.n_redouble = 0

        self.turn = self.bidding_seats[0]
        self.done_bidding = False

        self.tricks = np.zeros(NUM_PLAYERS)
        self.score_play = np.zeros(NUM_PLAYERS)

        # TODO[ス I've got no idea what the vars below do
        self.strain_declarer = {0: {}, 1: {}}
        self.group_declarer = -1
        if predeal_seats is None:
            predeal_seats = self.bidding_seats

        predeal = {}
        random.shuffle(self.cards)
        if reshuffle:  # generate new hands for predeal seats.
            i = 0
            self.one_hot_deal = np.zeros((len(Seat), len(FULL_DECK)), dtype=np.uint8)
            for seat in sorted(predeal_seats):
                predeal[seat] = self.cards[i: i+len(Rank)]
                self.one_hot_deal[seat] = one_hot_holding(predeal[seat]) # one hot cards
                i += len(Rank) # shift the index
            self.deal = Deal.prepare(predeal)

        if self.debug:
            convert_hands2string(self.deal)

        # setting the play elimination signals
        self.elimination_signal_play = 1 - self.one_hot_deal

        # if not allocated, zero vector is returned.
        return (self.one_hot_deal[self.turn], self.bidding_history), {"turn": Seat[self.turn], "max_bid": self.max_bid}

    def step_bid(self, action_bid):
        """
        This method performs a bidding action submitted via the 'action' argument, and performs an
        update of self.bidding_history and self.auction_history

        :param action_bid: bid action

        :return: state, reward, done_bidding
        """
        if self.done_bidding:
            raise Exception("No more actions can be taken")

        # action_bid must be in [0; AUCTION_SPACE_SIZE - 1]
        if action_bid < 0 or action_bid > AUCTION_SPACE_SIZE - 1:
            raise Exception("illegal action")

        # what happens when we get a pass
        if action_bid == PASS_IDX:
            self.bidding_history[action_bid] = 1 # PASS

            if self.max_bid == -1:
                self.auction_history[self.n_pass] = 1
            elif self.n_pass < 2:
                self.auction_history[
                    3 + 8*self.max_bid + 3*(self.n_double + self.n_redouble) + self.n_pass + 1] = 1

            # incrementing the current number of passes
            self.n_pass += 1
        # what happens when we get a contract bid
        elif action_bid < PASS_IDX:

            if action_bid <= self.max_bid:
                raise Exception("illegal bidding.")

            # resetting n_pass, n_double and n_redouble
            self.n_pass = 0
            self.n_double = 0
            self.n_redouble = 0
            self.max_bid = action_bid

            self.bidding_history[action_bid] = 1
            self.bidding_history[-1] = 0
            self.auction_history[3 + 8*self.max_bid] = 1

            # this action can no longer be performed
            self.elimination_signal[self.max_bid] = 1

            # doubles and redoubles are now permitted
            self.elimination_signal[REDOUBLE_RANGE] = 0

            strain = convert_action2strain(action_bid)
            group = Seat2Group[self.turn]
            if self.strain_declarer[group].get(strain, '') == '':
                self.strain_declarer[group][strain] = self.turn # which one
            self.group_declarer = group # which group
        # what happens when we get a double
        elif action_bid == DOUBLE_IDX:
            # doubles are not permitted when
            #    no contract bids have been made OR
            #    a double bid has already been made OR
            #    a redouble bid has been made
            if (self.max_bid == -1) or (self.n_double == 1) or (self.n_redouble == 1):
                raise Exception("double is not currently allowed")

            self.n_double = 1
            self.elimination_signal[DOUBLE_IDX] = 1
            self.auction_history[3 + 8*self.max_bid + 3] = 1
        # what happens when we get a redouble
        elif action_bid == REDOUBLE_IDX:
            # doubles are not permitted when
            #    no contract bids have been made OR
            #    a double bid has already been made
            if (self.max_bid == -1) or (self.n_redouble == 1):
                raise Exception("redouble is not currently allowed")

            self.n_redouble = 1
            self.elimination_signal[REDOUBLE_IDX] = 1
            self.auction_history[3 + 8*self.max_bid + 6] = 1

        # updating the ID of the next bidding player
        self.turn = (self.turn+1) % len(Seat)  # loop

        # move to the participant
        # TODO[ス: should not be used in the multi-agent version
        while True:
            if self.turn not in self.bidding_seats:
                self.turn = (self.turn+1) % len(Seat)
                self.n_pass += 1
            else:
                break

        hand = self.one_hot_deal[self.turn]
        reward = 0
        # state is the next bidding player's state
        if (self.n_pass >= 3 and self.max_bid < 0) or self.max_bid == 34:

            if self.max_bid < 0:
                raise Exception("illegal bidding")
            # extract the declarer, strain , level
            strain = convert_action2strain(self.max_bid)
            level = convert_action2level(self.max_bid)
            # single thread
            # reward = np.mean(Deal.score_st(dealer=self.deal, level=level, strain=strain, declarer=declarer, tries=self.nmc, mode=self.score_mode))
            # parallel threads

            # np.mean is moved to score
            declarer = self.strain_declarer[self.group_declarer][strain] # thise group's first declarer
            reward = Deal.score(dealer=self.deal,
                                level=level,
                                strain=strain,
                                declarer=declarer,
                                tries=self.nmc,
                                mode=self.score_mode)
            self.done_bidding = True

        state = (hand, self.bidding_history)
        info = {"turn": Seat[self.turn], "max_bid": self.max_bid}
        if self.debug:
            log_state(state, reward, self.done_bidding, info)

        return state, reward, self.done_bidding, info

    def score(self,
              tricks: Union[List, np.array],
              bid: int = None,
              vulnerability: Union[List, np.array] = None) -> np.array:
        """
        Calculates the score given the number of tricks, bid and vulnerability

        :param tricks: vector of trick counts
        :param bid:
        :param vulnerability:

        :return: an np.array of scores for all players
        """

        # using class's internal values if bid or vulnerability were not submitted
        # TODO[ス this will break if the class has just been initialised - please add more checks
        if bid is None:
            bid = self.max_bid
        if vulnerability is None:
            vulnerability = self.vulnerability

        out = np.zeros(NUM_PLAYERS)

        return out

    def step_play(self, action_play):

        """
        Performs a playing action

        :param action_play: a [1, 52] np.array with a single one

        :return:
        """

        # do nothing if we are not done bidding yet or we are done playing
        if (not self.done_bidding) or self.done_playing:
            pass

        # checking whether the player has the submitted action_play (card)

        return None
