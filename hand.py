from card import Card
import numpy as np


class Hand:
    # Constructor set the cards by the given list of cards, empty by default.
    def __init__(self, list=[]):
        # cards - list of Card objects, representing what's in the hand.
        self.cards = list

    # Remove a card from the hand, if it exists
    def removeCard(self, c):
        if c in self.cards:
            self.cards.remove(c)

    # Play a card in our hand, based on an index i
    def play(self, i):
        # Finds the card at index i, removes it from the hand and returns it
        return self.cards.pop(i)

    # Takes index i of card in valid list, and returns index in cards
    def validToRealIndex(self, i):
        c = self.validCards()[i]
        return self.cards.index(c)

    # For sorting the cards in hand
    def sort(self):
        self.cards.sort(reverse=True)

    # Returns a list of valid cards to be played, based on the lead suit. When
    # no lead suit has been established, Card.lead = -1.
    def validCards(self):
        lst = []

        for c in self.cards:
            if c.suit == Card.lead:
                lst.append(c)

        if len(lst) > 0:  # We had some cards of the lead suit
            return lst
        else:  # No cards of lead suit, so all are valid
            return self.cards

    # ----- PRINTING METHODS -----
    # String representation
    def __str__(self):
        return Card.printList(self.cards)

    def print_sorted(self):
        sorted_list = []
        for i in range(4):
            suit_cards = []
            for c in range(len(self.cards)):
                if self.cards[c].suit == i:
                    suit_cards.append(self.cards[c])
            suit_cards.sort(reverse=True)
            sorted_list = sorted_list + suit_cards
        return Card.printList(sorted_list)

    # ------ COUNTING METHODS -----
    # Count number of cards in each suit
    def suit_count(self):
        suit_ct = {i: 0 for i in range(4)}
        for c in self.cards:
            suit_ct[c.suit] += 1
        return suit_ct

    # Count number of aces in each suit
    def ace_by_suit(self):
        return self.card_ct(14)

    # Count number of kings in each suit
    def king_by_suit(self):
        return self.card_ct(13)

    # Count number of a specific card in each suit
    def card_ct(self, card_val):
        val_ct = {i: 0 for i in range(4)}
        for card in self.cards:
            if card.value == card_val:
                val_ct[card.suit] += 1
        return val_ct

    # Count number of Trump cards
    def trump_ct(self):
        return sum(card.trump == card.suit for card in self.cards)

    # ------ SUIT-BASED METHODS -----
    # Get all cards of a given suit
    def get_suit(self, suit):
        return [card for card in self.cards if card.suit == suit]

    # Get the largest cards from each suit
    def max_suit(self, suit):
        return max(self.get_suit(suit))

    # ------ DATA CONVERSION -----
    # Returns cards as a 52-length binary vector
    def get_cards_as_binary(self):
        x_binary = np.zeros(52).astype(int)
        for c in range(len(self.cards)):
            card = self.cards[c]
            value = card.value
            suit = card.suit
            x_binary[value + 4 * suit] = 1
        return x_binary

    # Returns cards as a 1-by-52 length binary vector (correct dimensions for
    # NN).
    def get_cards_binary(self, n):
        binvec = np.zeros((1, 52))
        for c in self.cards:
            binvec[0, c.suit * n + c.value - 2] = 1
        return binvec

    # Returns the cards as a list of values then suits, sorted in descending
    # order.
    def get_cards_as_val_suit_sorted(self):
        vals_sorted = np.zeros(13).astype(int)
        suits_sorted = np.zeros(13).astype(int)
        self.sort()
        for c in range(len(self.cards)):
            card = self.cards[c]
            value = card.value
            suit = card.suit
            vals_sorted[c] = value
            suits_sorted[c] = suit
        x_obs = np.concatenate([vals_sorted, suits_sorted])
        return x_obs

    # Returns the cards as a list of values then suits.
    def get_cards_as_val_suit(self):
        vals = np.zeros(13).astype(int)
        suits = np.zeros(13).astype(int)
        for c in range(len(self.cards)):
            card = self.cards[c]
            value = card.value
            suit = card.suit
            vals_sorted[c] = value
            suits_sorted[c] = suit
        x_obs = np.concatenate([vals, suits])
        return x_obs

    # Returns the cards as value-suit pairs in one list.
    def get_cards_as_interleave(self):
        vals = np.zeros(13).astype(int)
        suits = np.zeros(13).astype(int)
        for c in range(len(self.cards)):
            card = self.cards[c]
            value = card.value
            suit = card.suit
            vals_sorted[c] = value
            suits_sorted[c] = suit
        x_interleave = np.array([val for pair in zip(vals, suits) for val in pair]).astype(int)
        return x_interleave

    # Returns the cards as value-suit pairs in one list, sorted in descending
    # order.
    def get_cards_as_interleave_sorted(self):
        vals_sorted = np.zeros(13).astype(int)
        suits_sorted = np.zeros(13).astype(int)
        self.sort()
        for c in range(len(self.cards)):
            card = self.cards[c]
            value = card.value
            suit = card.suit
            vals_sorted[c] = value
            suits_sorted[c] = suit
        x_interleave_sorted = np.array([val for pair in zip(vals_sorted, suits_sorted) for val in pair]).astype(int)
        return x_interleave_sorted

    # Returns a 4-by-13 binary matrix with ones for cards existing in the hand
    def get_cards_as_matrix(self):
        binmat = np.zeros((4, 13))
        for c in self.cards:
            binmat[c.suit, c.value - 2] = 1
        return np.reshape(binmat, (4, 13, 1))

    # Returns a 52-bit binary in the same order as the matrix order
    def get_cards_matrix_order(self):
        binmat = np.zeros((4, 13))
        for c in self.cards:
            binmat[c.suit, c.value - 2] = 1
        return binmat
