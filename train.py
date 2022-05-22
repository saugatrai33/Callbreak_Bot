import numpy as np
import random as rnd
import datetime as dt
import keras

from game import Game
from ai_player import AIPlayer
from model import initialize_parameters, construct_bet_NN, construct_play_NN
from loss import batch_loss_history


# Creates AI objects to be used for playing games, with epsilon value depending
# on round t.
def makeAIs(t):
    AIs = [None for p in range(4)]
    # Make new AIPlayer objects to refresh its model
    for p in range(4):
        if strategies[p] == 4:
            AIs[p] = AIPlayer(strategies[p], bet_strategies[p], 'matrix', bet_model, action_model, None)
        elif bet_strategies[p] == 'model':
            AIs[p] = AIPlayer(strategies[p], bet_strategies[p], 'matrix', bet_models[p], None, None)
        else:
            AIs[p] = AIPlayer(strategies[p], bet_strategies[p], 'matrix', None, None, None)

    return AIs


# Copies the playing NN. The Target Action model will be used in calculating the training
# loss, to help ensure convergence and reduce oscillations in performance.
def copy_target(action_model):
    target_action_model = keras.models.clone_model(action_model)
    target_action_model.set_weights(action_model.get_weights())
    return target_action_model


# -------------------------------
#  INITIALIZATION OF VARIABLES
# -------------------------------
# Number of rounds of play to run
# num_tests = 100000
num_tests = 1

# Interval at which to train
# train_interval = 1000
train_interval = 1
# Offset of training for betting and playing
train_offset = train_interval / 2
# Number of samples to use for DQN training
train_batch_size = 100
# Memory buffer size of episodes to store
train_memory = 10000

# For tracking scores in the games
total_team1 = 0
total_team2 = 0
wins_team1 = 0.0
wins_team2 = 0.0
ties = 0.0

# Number of cards to give to each player, and number of tricks in each round
n = 13

# Discount factor for the reward
gamma = 0.99

# Strategies that each player should use to play
strategies = [4, 4, 4, 4]
bet_strategies = ['model', 'model', 'model', 'model']

# For saving the game state after each game
Hands = []
History = []
Bets = []
Scores = []
Tricks = []

# For tracking performance of the NN throughout training
Bet_Model_History = []
Play_Model_History = []
Average_Scores = []

# Save the current time
tempTime = dt.datetime.now().time()
timeString = str(dt.datetime.now().date()) + '-' + str(tempTime.hour) + '-' + str(tempTime.minute) + '-' + str(
    tempTime.second)

# ----------------------------------
#  INITIALIZATION OF NEURAL NETS
# ----------------------------------
# Initialize the parameters
[sgd, opt, batchsize, num_epochs, reg] = initialize_parameters()

bet_model = construct_bet_NN(n, True)
action_model = construct_play_NN(n, True)

target_action_model = copy_target(action_model)

datatype = 'matrix'
bet_models = [bet_model, bet_model, bet_model, bet_model]

# --------------------------
#  PLAY AND TRAIN
# --------------------------
# Initialize training data for NNs
x_train = []
y_train = []
x_train_RL = {'order': [], 'players': [], 'bets': [], 'tricks': [], 'hand': [], 'lead': []}
y_train_RL = []

AIs = makeAIs(0)

# Play the game for num_tests rounds
for t in range(1, num_tests + 1):
    # Count 100's of rounds
    if t % 100 == 0:
        print t

    # Play the game
    game = Game(n, strategies, bet_strategies, n, AIs)
    scores = game.playGame()

    # Save the scores for each team
    score_team1 = scores[0] + scores[2]
    score_team2 = scores[1] + scores[3]
    total_team1 += score_team1
    total_team2 += score_team2

    # Check who won the game
    if score_team2 < score_team1:
        wins_team1 += 1
    elif score_team2 > score_team1:
        wins_team2 += 1
    else:
        ties += 1

    # Get the initial hands of each player
    init_hands = [game.H_history[0][p] for p in range(4)]

    # Save data for betting training
    for p in range(4):
        # Save the hands as training data for the betting NN
        init_hands[p].sort()
        x_train.append(init_hands[p].get_cards_as_matrix())
        y_train.append(game.tricks[p])

    # Save the data from the game
    Hands.append(game.initialHands)
    History.append(game.h)
    Bets.append(game.bets)
    Scores.append(scores)
    Tricks.append(game.tricks)

    # Save the game state as training data for the playing NN
    for rd in range(n):
        # Get the game state up until this round
        round_state = game.action_state(rd)

        # Save data for training
        for p in range(4):
            # Make the state relative to player p
            state = game.relativize_state(round_state, p, rd)
            for key in state:
                x_train_RL[key].append(state[key])

            # Target Q value
            if rd == n:  # Final round in this episode reward is final total team score
                y_hat = game.tricks[p] + game.tricks[(p + 2) % 4]
                # y_hat = scores[p] + scores[(p+2)%4]
            else:  # More rounds left get reward based on target action
                y_hat = gamma * np.max(target_action_model.predict(game.potential_states(state, p, rd)))

            # Save this target label
            y_train_RL.append(y_hat)

    ############################################
    ############### DQN TRAINING ###############
    ############################################
    # Train the playing NN on a random sample of previous episodes
    if t > train_memory:
        # If we have more saved episodes than our memory allows, remove old data
        if len(y_train_RL) > train_memory:
            y_train_RL = y_train_RL[-train_memory:]
            y_train = y_train[-train_memory:]
            for key in x_train_RL:
                x_train_RL[key] = x_train_RL[key][-train_memory:]
                x_train[key] = x_train[key][-train_memory:]

        sample_inds = rnd.sample(range(len(x_train_RL['tricks'])), train_batch_size)
        x_sample = {}
        for key in x_train_RL:
            x_sample[key] = np.asarray([x_train_RL[key][s] for s in sample_inds])
        y_sample = [y_train_RL[s] for s in sample_inds]
        action_model.fit(x_sample, np.asarray(y_sample), batch_size=train_batch_size, epochs=1, verbose=0)

        # Make new AIPlayer objects to refresh models
        AIs = makeAIs(t)

    ################################################
    ############### BETTING TRAINING ###############
    ################################################
    # Train the betting NN
    if t % train_interval == 0:
        print 'Training betting...'
        # Train the NN
        hist = batch_loss_history()
        bet_model.fit(np.asarray(x_train), np.asarray(y_train), batch_size=batchsize, epochs=num_epochs, verbose=0,
                      callbacks=[hist])

        # Record the performance
        training_range = range(int(t / train_interval - 1) * train_interval, int(t / train_interval) * train_interval)
        Bet_Model_History.append(np.mean(hist.losses))  # Average loss across all batches
        Average_Scores.append([sum([Scores[i][p] for i in training_range]) / train_interval for p in
                               range(4)])  # Average score when using the previous strategy

        # Make new AIPlayer objects to refresh models
        AIs = makeAIs(t)

        print 'Done.'

    ######################################################
    ############### COPY TARGET Q FUNCTION ###############
    ######################################################
    # Copy the playing NN
    if (t + train_offset) % train_interval == 0:
        print 'Copying target action model...'
        target_action_model = copy_target(action_model)

        print 'Done.'

    # Save the models after they have been trained 10 times
    if t % (train_interval * 10) == 0:
        # Save the models
        action_model.save('./Models/action_' + timeString + '_' + str(t) + '.h5')
        bet_model.save('./Models/bet_' + timeString + '_' + str(t) + '.h5')

Total_Scores = [sum([Scores[i][p] for i in range(num_tests)]) for p in range(4)]
