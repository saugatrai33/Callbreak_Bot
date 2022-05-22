from deck import Deck
from card import Card

import random as rnd


# Pick a card to play from valid_cards, using a heuristic strategy.
def heuristicChoice(p, position, valid_cards, cards_played_by, cards_this_round, suit_trumped_by, bet_deficits):
    pos = position + 1
    partner = (p + 2) % 4
    leadsuit = valid_cards[0].lead
    trumpsuit = valid_cards[0].trump
    # split valid cards into lead suit, trump cards, and throwaways
    lead_cards = [card for card in valid_cards if card.suit == leadsuit]
    trump_cards = [card for card in valid_cards if card.suit == trumpsuit]
    throwaway_cards = [card for card in valid_cards if card.suit != Card.lead and card.suit != Card.trump]
    cur_max = current_max_card(cards_this_round)
    current_winner = currently_winning_player(cur_max, cards_this_round)
    chosen_card = choose_card(p, pos, cur_max, valid_cards, current_winner, lead_cards, trump_cards, throwaway_cards,
                              partner, bet_deficits, suit_trumped_by, cards_played_by)
    if chosen_card is None:
        return rnd.randint(0, len(valid_cards) - 1)
    chosen_card_valid_idx = None
    for i in range(0, len(valid_cards)):
        if str(chosen_card) == str(valid_cards[i]):
            chosen_card_valid_idx = i
    if chosen_card_valid_idx is None:
        print 'warning: chosen card index was none'
        print chosen_card_valid_idx
        print chosen_card
        print valid_cards
        return rnd.randint(0, len(valid_cards) - 1)

    return chosen_card_valid_idx


# Pick a card to play
def choose_card(p, pos, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits, suit_trumped_by,
                cards_played_by):
    if pos == 1:
        choice = move_first(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits,
                            suit_trumped_by, cards_played_by)
    elif pos == 2:
        choice = move_second(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits,
                             suit_trumped_by, cards_played_by)
    elif pos == 3:
        choice = move_third(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits,
                            suit_trumped_by, cards_played_by)
    elif pos == 4:
        choice = move_fourth(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits,
                             suit_trumped_by, cards_played_by)

    return choice


# ----- ORDERED PLAYING METHODS -----
# If the AI is going first
def move_first(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits, suit_trumped_by,
               cards_played_by):
    max_val = -1
    max_card = None
    choice = None
    for card in valid_cards:
        if card.value > max_val and card.suit != Card.trump:
            max_card = card
    choice = max_card
    if choice is None:
        choice = min(valid_cards)
    return choice


# If the AI is going second
def move_second(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits, suit_trumped_by,
                cards_played_by):
    choice = move_not_last(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits,
                           suit_trumped_by, cards_played_by)
    return choice


# If the AI is going third
def move_third(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits, suit_trumped_by,
               cards_played_by):
    choice = move_not_last(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits,
                           suit_trumped_by, cards_played_by)
    return choice


# If the AI is going fourth
def move_fourth(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits, suit_trumped_by,
                cards_played_by):
    choice = move_last(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits,
                       suit_trumped_by, cards_played_by)
    return choice


# ----- POSITION-BASED PLAYING METHODS -----
# If the AI isn't going last
def move_not_last(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits, suit_trumped_by,
                  cards_played_by):
    # find smallest card that wins, if it exists
    winnable = min_winnable(cur_max, suitc, trumpc)
    losable = min_throwable(suitc, throwc, trumpc)
    if winnable is not None:
        winnable_safe = determine_safe(cards_played_by, winnable, prtner_idx, suit_trumped_by)
    else:
        winnable_safe = 'N/A'
    # if partner is not winning, try to win if possible, or throw lowest card
    if winner != prtner_idx:
        if winnable is not None:
            if winnable_safe == True:
                return winnable
            else:
                return losable
        else:
            return losable
    # if partner is winning, try to win if partners deficit is less than own
    else:
        if bet_deficits[prtner_idx] < bet_deficits[p] and winnable is not None and winnable_safe == True:
            return winnable
        else:
            return losable


# If the AI is going last
def move_last(p, cur_max, valid_cards, winner, suitc, trumpc, throwc, prtner_idx, bet_deficits, suit_trumped_by,
              cards_played_by):
    # find smallest card that wins, if it exists
    winnable = min_winnable(cur_max, suitc, trumpc)
    losable = min_throwable(suitc, throwc, trumpc)
    # if partner is not winning, try to win if possible, or throw lowest card
    if winner != prtner_idx:
        if winnable is not None:
            choice = winnable
        else:
            choice = losable
    # if partner is winning, try to win if partners deficit is less than own
    else:
        if bet_deficits[prtner_idx] < bet_deficits[p] and winnable is not None:
            choice = winnable
        else:
            choice = losable
    return choice


# ----- HELPER METHODS -----
# The current highest card played this round
def current_max_card(cards_this_round):
    non_null_cards = [card for card in cards_this_round.values() if card is not None]
    if len(non_null_cards) > 0:
        return max(non_null_cards)
    else:
        return None


# Check which player is currently winning
def currently_winning_player(cur_max, cards_this_round):
    for i in range(4):
        if cards_this_round[i] is not None:
            if cards_this_round[i] == cur_max:
                return i
    else:
        return None


# Determine the AI's position in the turn order
def determine_position(p, cards_this_round):
    position = 4 - sum([cards_this_round[i] is None for i in range(4)]) + 1
    # print position
    return position


# Returns the HIGHEST card in the AI's hand of the lead suit
def max_lead_suit(valid_cards):
    return max([card for card in valid_cards if card.suit == card.lead])


# Returns the LOWEST card in the AI's hand of the lead suit
def min_lead_suit(valid_cards):
    return min([card for card in valid_cards if card.suit == card.lead])


# Returns the smallest-value valid card in the AI's hand of the lead suit that
# is better than other_card
def min_lead_suit_gt_card(valid_cards, other_card):
    winning_cards = [card for card in valid_cards if card.suit == card.lead and card > other_card]
    if len(winning_cards) > 0:
        return min(winning_cards)
    else:
        return None


# Finds the smallest card the AI could throw away. First looks in suits that
# match the lead suit, then from the throwaway suits, and lastly Trump.
def min_throwable(suitc, throwc, trumpc):
    if len(suitc) > 0:
        return min(suitc)
    if len(throwc) > 0:
        return min(throwc)
    else:
        return min(trumpc)


# Determine the minimum card that can win the current hand, if one exists
def min_winnable(max_card, suitc, trumpc):
    if len(suitc) > 0:
        return min_lead_suit_gt_card(suitc, max_card)
    if len(trumpc) > 0:
        return min(trumpc)
    else:
        return None


# Checks whether any card of a specific suit has been played
def suit_broken(cards_played, suit):
    for card in Deck().cards:
        if card.suit == suit:
            if cards_played[str(card)] is not None:
                return True
    return False


# Pick a random card from the valid cards
def choose_random(valid_cards):
    return valid_cards[rnd.randint(0, len(valid_cards) - 1)]


# Determine whether a card is 'safe' to play, i.e. not the case that higher of the lead suit are out there and not the case that suit is known to be trumped by opponents
def determine_safe(cards_played_by, card_considered, partner, suit_trumped_by):
    higher_cards = card_considered.get_higher_cards()
    trumpers = suit_trumped_by[card_considered.suit]
    # check if suit broken by opponents
    if len(trumpers.symmetric_difference([partner])) > 0:
        return False
    # if not broken, check whether higher cards of the same suit have not yet been played
    else:
        higher_of_suit = [card for card in higher_cards if
                          card.suit == card_considered.suit and cards_played_by[str(card)] is None]
        if len(higher_of_suit) > 0:
            return False
    return True


# Make a bet on how many tricks will be won
def heuristicBet(hand):
    bet = min(sum(hand.ace_by_suit().values()) + sum(hand.king_by_suit().values()) + round(hand.trump_ct() / 4), 13)
    return bet
