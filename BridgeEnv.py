from typing import List, Union, Tuple
from copy import deepcopy

from bridge_env.deal import Deal
from bridge_env.contract import Contract
from bridge_env.config import *
from bridge_env.bridge_utils import *
from bridge_env.score import precompute_scores_v2

import random
import numpy as np
import pandas as pd

# CHEAT-SHEET FOR BIDS
# 0-34: contract bids [1C, 1D, 1H, 1S, 1NT, ..., 7S, 7NT]
#   35: pass
#   36: double
#   37: redouble

# CHEAT-SHEET FOR keys of self._score_table:
#    (bid_tricks, trump, actual_tricks, vul, double)


class BridgeEnv(object):
    """
    This class is intended to replicate bridge bidding and playing
    """

    def __init__(self,
                 bidding_seats=Seat,
                 nmc=20,
                 debug=False,
                 score_mode="IMP"):

        # pre-calculating scores
        self._score_table = precompute_scores_v2(full_version=True)

        self.cards = deepcopy(FULL_DECK)

        # deal is the state
        self.deal = None
        self.one_hot_deal = None
        self.one_hot_known_deal = None
        self.vulnerability = None
        self.vulnerabilities = None

        self.auction_history = None

        self.history_bid = None
        self.history_play = None

        # bidding elimination signals - i.e. which bids are currently not permitted
        self.elim_sig_bid = None

        # playing elimination signals - i.e. which cards each player does not possess
        self.elim_sig_play = None

        self.n_pass = None
        self.n_double = None
        self.n_redouble = None

        self.nmc = nmc # MC times
        self.max_bid = None
        self.contract = Contract()
        self.done_bidding = None
        self.done_playing = None

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

        # index of the first bidder; start from the smallest one by default
        self.turn_bid = None

        # index of the first player
        self.turn_play = None

        # index of playing rounds complete
        self.n_play_actions = None
        self.n_play_turns = None

        # vector of tricks for all players
        self.tricks = None

        # matrix of current scores
        self.score_play = None

        # (re)setting the key environment variables upon initialisation
        self._reset()

    def _reset(self) -> None:
        """
        An internal method that resets the key variables held by the class for the new game

        :return: None
        """

        self.done_bidding = False
        self.done_playing = False

        self.contract.reset()

        # resetting bidding history
        # 1C 1D 1H 1S 1N ... 7N (PASS - not considered)
        self.history_bid = np.zeros(36, dtype=np.uint8)

        self.history_play = np.full((NUM_PLAYERS, NUM_PLAYS), np.nan)

        # generating vulnerabilities
        self.vulnerability = (np.random.rand(NUM_PAIRS) > 0.5).astype(int)
        self.vulnerabilities = self.vulnerability[[0, 1, 0, 1]]

        # resetting auction_history
        self.auction_history = np.zeros(self.auction_history_shape, dtype=np.uint8)

        # resetting bidding elimination signal - doubles and redoubles are not allowed at the start
        self.elim_sig_bid = np.ones(AUCTION_SPACE_SIZE, dtype=np.uint8)
        self.elim_sig_bid[REDOUBLE_RANGE] = 0

        # resetting the players' hands (aka one_hot_deal)
        self.one_hot_deal = np.zeros((NUM_PLAYERS, NUM_CARDS), dtype=np.uint8)
        self.one_hot_known_deal = np.ones((NUM_PLAYERS, NUM_CARDS), dtype=np.uint8)

        # resetting playing elimination signal: without
        self.elim_sig_play = np.ones((NUM_PLAYERS, NUM_CARDS), dtype=np.uint8)

        # resetting various counts
        self.max_bid = -1

        self.n_pass = 0
        self.n_double = 0
        self.n_redouble = 0

        # index of the first bidder; start from the smallest one by default
        # TODO[ス: should I care that the player in the 1st bidding seat always starts bidding?
        #  I don't see how this could cause any problems
        self.turn_bid = self.bidding_seats[0]
        self.turn_play = None

        self.n_play_actions = 0
        self.n_play_turns = 0
        self.tricks = np.zeros(NUM_PLAYERS, dtype=np.uint8)
        self.score_play = np.zeros((NUM_PLAYS + 1, NUM_PLAYERS))

    def _update_elim_sig_play(self) -> None:
        """
        An internal method that updates the play elimination signal: i.e. it recalculates the cards
        that each player has

        :return: None
        """

        self.elim_sig_play = self.one_hot_deal

    def _increment_n_play_actions(self) -> None:
        """
        An internal method that increments self.n_play_actions (the count of the number of actions
        that has been taken), and, if certain conditions are met, updates self.n_play_turns, the
        number of tricks taken and scores

        :return: None
        """

        self.n_play_actions += 1

        # incrementing the number of play turns that's been taken if all players have taken turn
        # at playing
        if self.n_play_actions % NUM_PLAYERS == 0:
            # updating the number of tricks taken (and hence scores)
            self._update_tricks_history()

        # setting self.done_playing if the maximum number of n_play_actions has been reached
        if self.n_play_actions == NUM_CARDS:
            self.done_playing = True

    def _update_tricks_history(self) -> None:

        # this is the most recent fully played-out round
        last_play = self.history_play[:, self.n_play_turns]

        # getting all the trump cards that were played in the most recent round (players
        # who did not play a trump card will be assigned the value 0)
        trump_cards = last_play * (last_play // 13 == self.contract.suit)

        # if the contract's suit is 'No Trump', or if trump_cards is empty
        # TODO[ス I am almost 100% positive that the two conditions are identical (iff), but I will
        #  keep it safe and explicit for the time being
        if (self.contract.suit_as_str == 'N') or (all(trump_cards == 0)):
            # getting the index of the player with the highest card in last_play
            winner = last_play.argmax()
        else:
            winner = trump_cards.argmax()

        # getting the indices of players on the winning side
        winners = Group2Seat[Seat2Group[winner]]

        # incrementing the tricks counter for the winning players
        self.tricks[winners] += 1

        # incrementing the number of playing turns
        self.n_play_turns += 1

        self._update_score()

    def reset(self,
              predeal_seats=None,
              reshuffle: bool = True,
              return_deal: bool = True):  # North and South
        """
        This method resets the environment - namely:
           - clears bidding history
           - generates new vulnerabilities
           - resets elimination signals (i.e. indicator of actions which cannot be performed)
           -

        :param predeal_seats: if not None, allocate cards to those seats. e.g. [0, 1] stands for
        North and East
        :param reshuffle: whether reshuffle the hands for the predeal seats
        :param return_deal: whether the newly generated deal should be returned or not

        :return: deal
        """

        self._reset()

        # TODO[ス I've got no idea what the vars and the code below do
        self.strain_declarer = {0: {}, 1: {}}
        self.group_declarer = -1
        if predeal_seats is None:
            predeal_seats = self.bidding_seats

        predeal = {}
        random.shuffle(self.cards)

        # generate new hands for predeal seats.
        if reshuffle:
            i = 0

            for seat in sorted(predeal_seats):
                predeal[seat] = self.cards[i: i+len(Rank)]

                # one hot cards
                self.one_hot_deal[seat] = one_hot_holding(predeal[seat])
                i += len(Rank) # shift the index
            self.deal = Deal.prepare(predeal)

        if self.debug:
            convert_hands2string(self.deal)

        # setting the play elimination signals
        self._update_elim_sig_play()

        if return_deal:
            # if not allocated, zero vector is returned.
            return (self.one_hot_deal[self.turn_bid], self.history_bid), \
                   {"turn": Seat[self.turn_bid], "max_bid": self.max_bid}
        else:
            pass

    def _update_score(self) -> None:
        """
        This method updates the current scores (from class's members), and puts them
        to the appropriate locations in self.score_play

        :return: None
        """

        # setting new score by iterating over players
        self.score_play[self.n_play_turns, ] = [
            self._score_table[(
                self.contract.level,
                self.contract.suit,
                self.tricks[i],
                self.contract.player_vulnerability[i],
                int(self.contract.double + self.contract.redouble)
            )]
            for i in range(NUM_PLAYERS)
        ]

    def step_bid(self, action_bid):
        """
        This method performs a single bid action submitted via the 'action' argument, and
        performs an update of self.history_bid and self.auction_history

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

            # we are not allowed to make a double for now
            self.elim_sig_bid[DOUBLE_IDX] = 0

            self.history_bid[action_bid] = 1

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

            self.history_bid[action_bid] = 1
            self.history_bid[-1] = 0
            self.auction_history[3 + 8*self.max_bid] = 1

            # this action and all the actions below can no longer be performed
            self.elim_sig_bid[:(1 + self.max_bid)] = 0

            # doubles are now permitted, redoubles are not permitted
            self.elim_sig_bid[DOUBLE_IDX] = 1
            self.elim_sig_bid[REDOUBLE_IDX] = 0

            strain = convert_action2strain(action_bid)
            group = Seat2Group[self.turn_bid]
            if self.strain_declarer[group].get(strain, '') == '':
                self.strain_declarer[group][strain] = self.turn_bid # which one
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
            self.elim_sig_bid[DOUBLE_IDX] = 0
            self.elim_sig_bid[REDOUBLE_IDX] = 1
            self.auction_history[3 + 8*self.max_bid + 3] = 1

        # what happens when we get a redouble
        elif action_bid == REDOUBLE_IDX:
            # redoubles are not permitted when
            #    no contract bids have been made OR
            #    a double bid has not been made OR
            #    a redouble bid has already been made
            if (self.max_bid == -1) or (self.n_double == 0) or (self.n_redouble == 1):
                raise Exception("redouble is not currently allowed")

            self.n_redouble = 1
            self.elim_sig_bid[DOUBLE_IDX] = 0
            self.elim_sig_bid[REDOUBLE_IDX] = 0
            self.auction_history[3 + 8*self.max_bid + 6] = 1

        # updating the index of the next bidding player
        self.turn_bid = (self.turn_bid + 1) % len(Seat)

        # move to the participant
        # NB: this code is only useful if not all players are bidding (i.e. self.bidding_seats
        # does not contain everybody)
        while True:
            if self.turn_bid not in self.bidding_seats:
                self.turn_bid = (self.turn_bid + 1) % len(Seat)
                self.n_pass += 1
            else:
                break

        hand = self.one_hot_deal[self.turn_bid]
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

            # TODO[ス: game rewards / scores will no longer be calculated during bidding - the next
            #  bit of code needs to be removed
            reward = Deal.score(dealer=self.deal,
                                level=level,
                                strain=strain,
                                declarer=declarer,
                                tries=self.nmc,
                                mode=self.score_mode)
            self.done_bidding = True

            # storing the contract
            self.contract.from_bid(bid=self.max_bid,
                                   double=(self.n_double > 0),
                                   redouble=(self.n_redouble > 0))

            # setting the index of the first player
            self.turn_play = (self.turn_bid + 1) % len(Seat)

            # since bidding is now done, we need to set the initial value of self.score_play
            self._update_score()

        # TODO[ス: remove the next lines - this method should no longer return anything
        state = (hand, self.history_bid)
        info = {"turn": Seat[self.turn_bid], "max_bid": self.max_bid}
        if self.debug:
            log_state(state, reward, self.done_bidding, info)

        return state, reward, self.done_bidding, info

    # TODO[ス deprecate self.score() method
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
        if bid is None:
            bid = self.max_bid
        if vulnerability is None:
            vulnerability = self.vulnerability

        out = np.zeros(NUM_PLAYERS)

        return out

    def step_play(self,
                  player: int,
                  action_play: int) -> None:
        """
        Performs a playing action:
           - storing the player's action to history_play
           - (if necessary) updating n_play_turns
           - (if necessary) updating scores

        :param player: a the index of the player that performs the action
        :param action_play: a [1, 52] np.array with a single one

        :return: None
        """

        # do nothing if we are not done bidding yet or we are done playing
        if (not self.done_bidding) or self.done_playing:
            pass

        # exception if the player does not exist
        if player not in Seat:
            raise Exception(f"Player {player} does not exist")

        # exception if the action is not in the range of [0, 51]
        if action_play not in range(0, NUM_CARDS):
            raise Exception(f"Action {action_play} is invalid")

        # exception if the player does not have the card (action_play) it wants to play
        if self.elim_sig_play[player, action_play] == 0:
            raise Exception(f"Player {player} does not posses the card {action_play}")

        # adding the current action to the play history
        self.history_play[player, self.n_play_turns] = action_play

        # updating the hand (one_hot_deal)
        self.one_hot_deal[player, action_play] = 0
        self.one_hot_known_deal[[x for x in range(NUM_PLAYERS) if x != player], action_play] = 0

        # updating the play elimination signal
        self._update_elim_sig_play()

        # this method will update tricks and scores (if it's necessary)
        self._increment_n_play_actions()

        # setting the index of the next player
        self.turn_play = (self.turn_play + 1) % len(Seat)

        return None

    def player_known_deal(self, player: int):

        out = self.one_hot_known_deal.copy()
        out[player, :] = self.one_hot_deal[player, :]

        idx = np.where(out[player, :] == 1)

        for x in range(NUM_PLAYERS):
            if x != player:
                out[x, idx] = 0

        return out

    @property
    def auction_history_shape(self) -> Tuple:
        """

        :return: A tuple of the shape of auction history vector
        """
        return (AUCTION_HISTORY_SIZE, )

    @property
    def deal_shape(self) -> Tuple:
        """

        :return: A tuple of the shape of the deal (i.e. cards held by all players)
        """

        return NUM_PLAYERS, NUM_CARDS

    @property
    def vuln_shape(self) -> Tuple:
        """

        :return: A tuple of the shape of the vulnerabilities vector
        """

        return (NUM_PLAYERS, )

    @property
    def belief_shape(self) -> Tuple:
        """

        :return:
        """
        return self.deal_shape

    @property
    def bid_policy_shape(self) -> Tuple:
        """

        :return:
        """
        return (AUCTION_SPACE_SIZE, )

    @property
    def play_policy_shape(self) -> Tuple:
        """

        :return:
        """

        return (NUM_CARDS, )

    @property
    def elim_sig_bid_shape(self) -> Tuple:
        """
        Shape of elimination signal for bidding actions (bids)

        :return: A tuple
        """
        return self.bid_policy_shape

    @property
    def elim_sig_play_shape(self) -> Tuple:
        """
        Shape of elimination signal for playing actions

        :return: A tuple
        """

        return self.play_policy_shape
