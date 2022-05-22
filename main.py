import keras

from game import Game
from loss import loss_bet, get_loss_bet

# Number of full games to play
num_games = 1
datatype = 'matrix'
bet_strategies = ['model', 'model', 'model', 'model']

hvh = keras.models.load_model('Models/Heuristic_v_Heuristic_bet_data_model_' + datatype + '_model.h5',
                              custom_objects={'get_loss_bet': get_loss_bet, 'loss_bet': loss_bet})

# To specify which playing NN to load
timestr = '2022-04-20-14-30-55'
iterations = 40000

nn_action_model = keras.models.load_model('Models/action_' + timestr + '_' + str(iterations) + '.h5',
                                          custom_objects={'get_loss_bet': get_loss_bet, 'loss_bet': loss_bet})
nn_bet_model = keras.models.load_model('Models/bet_' + timestr + '_' + str(iterations) + '.h5',
                                       custom_objects={'get_loss_bet': get_loss_bet, 'loss_bet': loss_bet})


# bet_strat = AIPlayer(4, 'model', 'matrix', nn_bet_model, None, None)


def test_game():
    strategies = [4, 3, 3, 3]
    bet_strategies = ['model', 'model', 'model', 'model']
    action_models = [nn_action_model, None, None, None]
    bet_models = [nn_bet_model, hvh, nn_bet_model, hvh]

    n = 13

    for i in range(1):
        for g in range(num_games):
            game = Game(n, strategies, bet_strategies, n, [None, None, None, None], action_models, bet_models)
            scores = game.playGame()
            print "Player1 Score -> %s", scores[0]
            print "Player2 Score -> %s", scores[1]
            print "Player3 Score -> %s", scores[2]
            print "Player4 Score -> %s", scores[3]


if __name__ == "__main__":
    test_game()
