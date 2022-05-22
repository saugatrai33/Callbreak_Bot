import random as rnd
import numpy as np

import custom_logic as hai


class AIPlayer:
    # Constructor
    def __init__(self, strategy, bettype, datatype, bet_object=None, action_object=None,
                 eps=0.05):
        self.strategy = strategy
        self.bettype = bettype
        self.datatype = datatype
        self.eps = eps
        self.action_model = action_object
        self.betmodel = bet_object

    # ----- PRINTING METHODS -----
    # String representation
    def __str__(self):
        return str(self.strategy.name)

    # ----- Get Action -----
    # Returns the index of the selected action, from the list of Cards 'actions'
    def get_action(self, n, p, state, actions, current_winner):
        if self.strategy == 4:
            # Greedily exploit current value function w.p. 1 - eps
            if self.eps == 0 or rnd.random() >= self.eps:
                potential_states = self.get_potential_states(n, p, state, actions, current_winner)
                values = self.action_model.predict(potential_states)
                # for x in range(len(values)):
                #    print str(actions[x]) + ': ' + str(values[x])
                ind = np.argmax(values)
            # Take random action w.p. eps
            else:
                ind = rnd.randint(0, len(actions) - 1)
        # Simple heuristic
        elif self.strategy == 3:
            ind = hai.heuristicChoice(p, state[0], actions, state[1], state[2], state[3], state[4])
        # Myopic Greedy: pick the highest playable card every time
        elif self.strategy == 2:
            ind = np.argmax(actions)
        # Random choice
        else:
            # Pick a valid card at random
            ind = rnd.randint(0, len(actions) - 1)
        return ind

    # ----- Get Bet -----
    def get_bet(self, hand):
        if self.bettype == 'model':
            model_bet = self.betmodel.predict(np.reshape(hand.get_cards_as_matrix(), (1, 4, 13, 1)))
            bet = max(min(13, np.round(model_bet)), 2)
        elif self.bettype == 'heuristic':
            bet = hai.heuristicBet(hand)
        else:
            bet = rnd.randint(2, 5)

        return bet

    # ------ Get Cards ----
    def get_cards(self, hand):
        if self.datatype == 'sorted':
            return hand.get_cards_as_val_suit_sorted()
        elif self.datatype == 'binary':
            return hand.get_cards_as_binary()
        elif self.datatype == 'interleave':
            return hand.get_cards_as_interleave()
        elif self.datatype == 'interleave_sorted':
            return hand.get_cards_as_interleave_sorted()
        elif self.datatype == 'standard':
            return hand.get_cards_as_val_suit()
        elif self.datatype == 'matrix':
            return hand.get_cards_as_matrix()

    # Returns a dictionary, where each value is a numpy array of arrays
    # corresponding to what the state would look like after an action was taken,
    # for each action in the list actions.
    def get_potential_states(self, n, p, state, actions, current_winner):
        # Find number of actions
        alen = len(actions)
        # Find number of cards that have been played so far
        count = np.max(state['order'][0])
        # Initialize an empty dictionary to empty arrays
        data = {}
        for key in state:
            data[key] = np.zeros((alen,) + state[key].shape)
        # Check if we would be setting the lead suit
        if state['lead'] == -1:
            change_lead = True
        else:
            change_lead = False

        for j in range(alen):
            # Get current action being considered
            a = actions[j]
            # Build the potential state after the action is taken
            aind = a.suit * n + a.value - 2
            state['order'][0, aind] = count + 1
            state['players'][0, aind] = 1
            state['hand'][0, aind] = 0
            if change_lead:  # The lead was changed
                state['lead'] = a.suit
            if (count + 1) % 4 == 0 and current_winner is not None:  # Update tricks
                if a > current_winner:  # If this would win
                    pwin = 0
                else:  # The current winner won
                    pwin = int(state['players'][0, current_winner.suit * n + current_winner.value - 2] - 1)
                state['tricks'][0, pwin] += 1
            # Add to the list
            for key in state:
                data[key][j,] = state[key]
            # Remove the potential state
            state['order'][0, aind] = 0
            state['players'][0, aind] = 0
            state['hand'][0, aind] = 1
            if change_lead:
                state['lead'] = -1
            if (count + 1) % 4 == 0 and current_winner is not None:  # Update tricks
                state['tricks'][0, pwin] -= 1

        return data
