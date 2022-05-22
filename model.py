import keras
import keras.layers as L
import keras.regularizers as R
import keras.models as M

from loss import get_loss_bet


# Initialize the NN optimizer and other parameters
def initialize_parameters():
    sgd = keras.optimizers.SGD(lr=.01, clipnorm=1.)
    opt = keras.optimizers.RMSprop(lr=.01, clipnorm=1.)
    batchsize = 128
    num_epochs = 20
    reg = 0.1
    return [sgd, opt, batchsize, num_epochs, reg]


# Construct the NN for betting
def construct_bet_NN(n, show_summary=False):
    # Initialize the parameters
    [sgd, opt, batchsize, num_epochs, reg] = initialize_parameters()

    # Initialize the betting NN
    input_layer = L.Input(shape=(4, n, 1))
    xinp = L.LeakyReLU()(input_layer)
    x = L.Conv2D(1, 1, activation='linear', use_bias=False, kernel_regularizer=R.l2(reg))(xinp)
    x = L.BatchNormalization(axis=1)(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(1, 1, activation='linear', use_bias=False, kernel_regularizer=R.l2(reg))(x)
    x = L.BatchNormalization(axis=1)(x)
    x = L.Add()([x, xinp])
    x = L.LeakyReLU()(x)
    x = L.Conv2D(2, (1, 4), activation='linear', use_bias=False, kernel_regularizer=R.l2(reg))(x)
    x = L.BatchNormalization(axis=1)(x)
    x = L.LeakyReLU()(x)
    x = L.Flatten()(x)
    output_layer = L.Dense(1, activation='linear', use_bias=False)(x)

    # Compile the model
    bet_model = M.Model(inputs=input_layer, outputs=output_layer)
    bet_model.compile(loss=get_loss_bet(), optimizer=sgd, metrics=['mean_absolute_error', get_loss_bet()])

    # Want to see a summary of the NN
    if show_summary:
        bet_model.summary()

    return bet_model


# Construct the NN for playing
def construct_play_NN(n, show_summary=False):
    # Initialize the parameters
    [sgd, opt, batchsize, num_epochs, reg] = initialize_parameters()

    # Initialize the playing NN branches
    input_order = L.Input(shape=(1, n * 4), name='order')
    input_players = L.Input(shape=(1, n * 4), name='players')
    input_hand = L.Input(shape=(1, n * 4), name='hand')
    input_bets = L.Input(shape=(1, 4), name='bets')
    input_tricks = L.Input(shape=(1, 4), name='tricks')

    # Branch for the history of the order of played cards
    xo = L.Reshape((4, 13, 1))(input_order)
    xo = L.Conv2D(1, 1, activation='linear', use_bias=False, kernel_regularizer=R.l2(reg))(xo)
    xo = L.BatchNormalization(axis=1)(xo)
    xo = L.LeakyReLU()(xo)
    xo = L.Conv2D(2, (1, 4), activation='linear', use_bias=False, kernel_regularizer=R.l2(reg))(xo)
    xo = L.BatchNormalization(axis=1)(xo)
    xo = L.LeakyReLU()(xo)
    xo = L.Flatten()(xo)

    # Branch for the history of which player played each card
    xp = L.Reshape((4, 13, 1))(input_players)
    xp = L.Conv2D(1, 1, activation='linear', use_bias=False, kernel_regularizer=R.l2(reg))(xp)
    xp = L.BatchNormalization(axis=1)(xp)
    xp = L.LeakyReLU()(xp)
    xp = L.Conv2D(2, (1, 4), activation='linear', use_bias=False, kernel_regularizer=R.l2(reg))(xp)
    xp = L.BatchNormalization(axis=1)(xp)
    xp = L.LeakyReLU()(xp)
    xp = L.Flatten()(xp)

    # Branch for the current player's hand
    xh = L.Reshape((4, 13, 1))(input_hand)
    xh = L.Conv2D(1, 1, activation='linear', use_bias=False, kernel_regularizer=R.l2(reg))(xh)
    xh = L.BatchNormalization(axis=1)(xh)
    xh = L.LeakyReLU()(xh)
    xh = L.Conv2D(2, (1, 4), activation='linear', use_bias=False, kernel_regularizer=R.l2(reg))(xh)
    xh = L.BatchNormalization(axis=1)(xh)
    xh = L.LeakyReLU()(xh)
    xh = L.Flatten()(xh)

    # Branch for the bets
    xb = L.LeakyReLU()(input_bets)
    xb = L.Flatten()(xb)

    # Branch for the tricks
    xt = L.LeakyReLU()(input_tricks)
    xt = L.Flatten()(xt)

    # Add all NN branches together
    a = L.Concatenate(axis=1)([xo, xp, xh, xb, xt])
    a = L.LeakyReLU()(a)
    output_a = L.Dense(1, activation='linear', use_bias=False, kernel_regularizer=R.l2(reg))(a)

    # Compile the model
    action_model = M.Model(inputs=[input_order, input_players, input_hand, input_bets, input_tricks], outputs=output_a)
    action_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_absolute_error'])

    # Want to see a summary of the NN
    if show_summary:
        action_model.summary()

    return action_model
