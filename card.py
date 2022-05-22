import numpy as np


class Card:
    # ----- STATIC CLASS VARIABLES -----
    # lead:     current lead suit
    # trump:    current trump suit
    lead = -1  # default lead suit is none
    trump = 3  # default trump suit is Spade

    # ----- OBJECT VARIABLES -----
    # value: value of the card. 11 = Jack, 12 = Queen, 13 = King, 14 = Ace.
    # suit:  suit of the card. 0 = Clubs, 1 = Diamonds, 2 = Hearts, 3 = Spades.

    # Constructor default is Ace of Spades.
    def __init__(self, v=1, s=3):
        self.value = v
        self.suit = s

    # ----- PRINTING METHODS -----
    # String representation
    def __str__(self):
        # First add the value, changing to a named card if necessary (ex. 1 -> A)
        if self.value == 1:
            string = 'A'
        elif self.value == 11:
            string = 'J'
        elif self.value == 12:
            string = 'Q'
        elif self.value == 13:
            string = 'K'
        else:
            string = str(self.value)

        # Add the suit
        if self.suit == 0:
            string += 'C'
        elif self.suit == 1:
            string += 'D'
        elif self.suit == 2:
            string += 'H'
        elif self.suit == 3:
            string += 'S'

        return string

    # For printing lists of cards
    @staticmethod
    def printList(cards):
        string = '<'

        for c in cards:
            string += str(c) + ', '

        return string[0:-2] + '>'

    # For printing lists of cards, with their indices
    @staticmethod
    def printListIndices(cards):
        string = ''

        for i in range(len(cards)):
            string += str(i) + ': ' + str(cards[i]) + ', '

        return string[0:-2]

    # Returns all cards that have a higher value than itself.
    def get_higher_cards(self):
        higher_cards = []

        for i in range(self.value + 1, 15):
            higher_cards.append(Card(i, self.suit))
        if self.suit != Card.trump:
            for i in range(2, 15):
                higher_cards.append(Card(i, self.trump))
        return higher_cards

    # Returns itself as a 1-by-n*4 binary vector with a 1 for this card.
    def as_action(self, n):
        bin_card = [0 for i in range(n * 4)]
        bin_card[(self.value - 1) + self.suit * n - 1] = 1
        return np.reshape(bin_card, (1, n * 4))

    # ----- COMPARISON METHODS -----
    # Test for card equality
    def __eq__(self, other):
        return self.suit == other.suit and self.value == other.value

    # Test for card inequality
    def __ne__(self, other):
        return not (self == other)

    # Test if I'm greater than other
    def __gt__(self, other):
        if self.suit == Card.trump:  # I'm trump
            if other.suit == Card.trump:  # We're both trump
                return self.value > other.value
            else:  # I'm trump, they're not
                return True
        elif self.suit == Card.lead:  # I'm the lead suit
            if other.suit == Card.trump:  # They're trump, I'm not
                return False
            elif other.suit == Card.lead:  # We're both the lead suit
                return self.value > other.value
            else:  # I'm the lead suit, they're not
                return True
        else:  # I'm not trump or the lead suit
            if other.suit == Card.trump or other.suit == Card.lead:  # They are
                return False
            else:  # Neither of us are trump or the lead suit
                return self.value > other.value

    # Test if I'm greater than or equal to other
    def __ge__(self, other):
        if self.suit == Card.trump:  # I'm trump
            if other.suit == Card.trump:  # We're both trump
                return self.value >= other.value
            else:  # I'm trump, they're not
                return True
        elif self.suit == Card.lead:  # I'm the lead suit
            if other.suit == Card.trump:  # They're trump, I'm not
                return False
            elif other.suit == Card.lead:  # We're both the lead suit
                return self.value >= other.value
            else:  # I'm the lead suit, they're not
                return True
        else:  # I'm not trump or the lead suit
            if other.suit == Card.trump or other.suit == Card.lead:  # They are
                return False
            else:  # Neither of us are trump or the lead suit
                return self.value >= other.value

    # Test if I'm less than other
    def __lt__(self, other):
        return not (self >= other)

    # Test if I'm less than or equal to other
    def __le__(self, other):
        return not (self > other)
