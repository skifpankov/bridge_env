from deal import Deal
from copy import deepcopy
from config import *
import random
import numpy as np
from bridge_utils import *

# CHEAT-SHEET FOR BIDS
# 0-34: contract bids [1C, 1D, 1H, 1S, 1NT, ..., 7S, 7NT]
#   35: pass
#   36: double
#   37: redouble


class BridgeEnv(object):
    def __init__(self, bidding_seats=[0, 2], nmc=20, debug=False, score_mode="IMP"):
        # deal is the state
        self.deal = None
        self.one_hot_deal = None
        self.cards = deepcopy(FULL_DECK)
        self.bidding_history = np.zeros(36, dtype=np.uint8) # pass is included in the history
        self.auction_history = np.zeros(AUCTION_HISTORY_SIZE, dtype=np.uint8)
        self.vulnerability = np.zeros(NUM_PAIRS, dtype=np.uint8)
        # vector of elimination signals - i.e. which actions are currently not permitted
        self.elimination_signal = np.zeros(AUCTION_SPACE_SIZE, dtype=np.uint8)

        self.n_pass = 0
        self.n_double = 0
        self.n_redouble = 0

        self.nmc = nmc # MC times
        self.max_bid = -1
        self.done = False
        self.debug = debug
        self.score_mode = score_mode
        self.strain_declarer = {0: {}, 1: {}}
        self.group_declarer = -1

        self.bidding_seats = sorted(list(set(bidding_seats)))
        for seat in self.bidding_seats:
            if seat not in Seat:
                raise Exception("illegal seats")
        self.turn = self.bidding_seats[0] # whose turn, start from the smallest one by default.

        # TODO: check whether resetting the environment on initialisation can break anything
        # resetting the environment upon initialisation
        # self.reset()

    def set_nmc(self, n):
        self.nmc = n

    def set_mode(self, debug):
        self.debug = debug

    def reset(self, predeal_seats=None, reshuffle=True):  # North and South
        """
        :param predeal_seats: if not None, allocate cards to those seats. e.g. [0, 1] stands for North and East
        :param reshuffle: whether reshuffle the hands for the predeal seats
        :return: deal
        """
        self.bidding_history = np.zeros(36, dtype=np.uint8) # 1C 1D 1H 1S 1N ... 7N (PASS - not considered)

        # generating vulnerabilities
        self.vulnerability = (np.random.rand(NUM_PAIRS) > 0.5).astype(int)

        # resetting auction_history
        self.auction_history = 0 * self.auction_history

        # resetting elimination_history - doubles and redoubles are not allowed at the start
        self.elimination_signal[REDOUBLE_RANGE] = 1

        self.max_bid = -1
        self.n_pass = 0
        self.n_double = 0
        self.n_redouble = 0

        self.turn = self.bidding_seats[0] # the first one.
        self.done = False
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

        # if not allocated, zero vector is returned.
        return (self.one_hot_deal[self.turn], self.bidding_history), {"turn": Seat[self.turn], "max_bid": self.max_bid}

    def step(self, action):
        """
        :param action: bid action
        :return: state, reward, done
        """
        if self.done:
            raise Exception("No more actions can be taken")

        # action must be in [0; AUCTION_SPACE_SIZE - 1]
        if action < 0 or action > AUCTION_SPACE_SIZE - 1:
            raise Exception("illegal action")

        # what happens when we get a pass
        if action == PASS_IDX:
            self.bidding_history[action] = 1 # PASS

            if self.max_bid == -1:
                self.auction_history[self.n_pass] = 1
            elif self.n_pass < 2:
                self.auction_history[
                    3 + 8*self.max_bid + 3*(self.n_double + self.n_redouble) + self.n_pass + 1] = 1

            # incrementing the current number of passes
            self.n_pass += 1
        # what happens when we get a contract bid
        elif action < PASS_IDX:

            if action <= self.max_bid:
                raise Exception("illegal bidding.")

            # resetting n_pass, n_double and n_redouble
            self.n_pass = 0
            self.n_double = 0
            self.n_redouble = 0
            self.max_bid = action

            self.bidding_history[action] = 1
            self.bidding_history[-1] = 0
            self.auction_history[3 + 8*self.max_bid] = 1

            # this action can no longer be performed
            self.elimination_signal[self.max_bid] = 1

            # doubles and redoubles are now permitted
            self.elimination_signal[REDOUBLE_RANGE] = 0

            strain = convert_action2strain(action)
            group = Seat2Group[self.turn]
            if self.strain_declarer[group].get(strain, '') == '':
                self.strain_declarer[group][strain] = self.turn # which one
            self.group_declarer = group # which group
        # what happens when we get a double
        elif action == DOUBLE_IDX:
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
        elif action == REDOUBLE_IDX:
            # doubles are not permitted when
            #    no contract bids have been made OR
            #    a double bid has already been made
            if (self.max_bid == -1) or (self.n_redouble == 1):
                raise Exception("redouble is not currently allowed")

            self.n_redouble = 1
            self.elimination_signal[REDOUBLE_IDX] = 1
            self.auction_history[3 + 8*self.max_bid + 6] = 1

        self.turn = (self.turn+1) % len(Seat)  # loop
        while True:  # move to the participant
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
            reward = Deal.score(dealer=self.deal, level=level, strain=strain, declarer=declarer, tries=self.nmc, mode=self.score_mode)
            self.done = True



        state = (hand, self.bidding_history)
        info = {"turn": Seat[self.turn], "max_bid": self.max_bid}
        if self.debug:
            log_state(state, reward, self.done, info)
        return state, reward, self.done, info
