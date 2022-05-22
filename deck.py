import random as rnd
from card import Card


class Deck:
    # Constructor make all possible cards in the deck, assuming a max value of n (default is 13).
    def __init__(self, n=13):
        self.cards = []

        # Add all cards to the deck
        for s in range(0, 4):
            for v in range(2, n + 2):
                self.cards.append(Card(v, s))

    # Shuffles the deck
    def shuffle(self):
        rnd.shuffle(self.cards)

    # Deal n cards, default 1.
    def deal(self, n=1):
        dealt = self.cards[0:n]
        del self.cards[0:n]
        return dealt

    # ----- PRINTING METHODS -----
    # String representation
    def __str__(self):
        return Card.printList(self.cards)
