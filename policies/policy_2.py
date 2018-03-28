#### general stuff ####
from __future__ import print_function
from policies import base_policy as bp
import numpy as np
import pickle
import sandbox

#### keras stuff ####

# TODO: i removed this 3 imports, which didn't work, and seemed useless so SOMETHING works
# from keras import initializations
# from keras.initializations import normal, identity
# from keras.models import model_from_json
from keras.models import Sequential, save_model,load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam

#### globals ####
GAME = 'flappy snake'  # the name of the game being played for log files
CONFIG = 'nothreshold'
AC_DIM = 3  # number of valid actions
AC_MAX = 2
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 3200.  # timesteps to observe before training
EXPLORE = 3000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 500  # number of previous transitions to remember
BATCH = 1  # size of minibatch
FRAME_PER_ACTION = 1
CHANNELS = 1


class Policy2(bp.Policy):

    def __init__(self, policy_args, board_size, stateq, actq, modelq, logq, id):
        super().__init__(policy_args, board_size, stateq, actq, modelq, logq, id)
        self.last_act = None
        self.EPSILON = 1
        self.board_size = board_size
        self.board_size_flat = board_size[0]*board_size[1]
        self.D = np.zeros((REPLAY_MEMORY, self.board_size_flat + 1 + 1 + self.board_size_flat))
        self.D_prime = []

    def cast_string_args(self, policy_args):
        # policy_args['load_from'] = policy_args['load_from'] if 'load_from' in policy_args else None
        return policy_args

    def init_run(self):
        # if self.load_from is not None:
        #     self.model = load_model(self.load_from)
        # else:
        self.model = self.buildmodel()
        self.r_sum = 0

    def learn(self, reward, t):
        if t>0:
            self.D[t%REPLAY_MEMORY-1,self.board_size_flat+1] = reward # r_t-1

    def get_batch(self):
        batch = np.zeros((BATCH, CHANNELS, self.board_size_flat))
        sample_ind = np.random.choice(range(REPLAY_MEMORY), replace=False, size=BATCH)
        samples = self.D[sample_ind]
        for i, ii in enumerate(sample_ind):
            batch[ii] = np.vstack([self.D[i-j] for j in range(CHANNELS)])
        return batch

    def gen_labels(self, batch):
        input_s = batch[:, :board_size_flat]
        fp_s = self.model.predict(input_s, batch_size = BATCH)
        input_s_prime = batch[:, -board_size_flat:]
        fp_s_prime = self.model.predict(input_s_prime, batch_size = BATCH)
        labels = fp_s
        labels[self.D[self.board_size_flat]] = self.D[self.board_size_flat+1] + GAMMA*np.max(fp_s_prime, axis=1)
        return labels

    def eps_sched(self):
        pass    

    def train_net(self):
        batch = self.get_batch()
        labels_s = gen_labels(self, batch)
        input_s = batch[:, :board_size_flat]
        model.fit(input_s, labels_s, batch_size=BATCH, nb_epoch=1, verbose=1)

    def act(self, t, state, player_state):
        if t>0:
            # we finish collecting  t-1=k:<s,a,r,s'> here
            self.D[t%REPLAY_MEMORY-1,-self.board_size_flat:] = state.flatten() # s'_t-1           

        # we start collecting t=k:<s,a,r,s'> here 
        self.D[t%REPLAY_MEMORY,:self.board_size_flat] = state.flatten() # s_t
        # compute step 2 in nervana pseudocode
        # backprop error
        p = np.random.randint(low=0, high=int(1./self.EPSILON*AC_DIM))
        print ('p', p)
        #self.last_act = self.ACTIONS[self.Q_sa(state)]
        net_input = np.vstack([self.D[(i-j) % REPLAY_MEMORY, :board_size_flat] for j in range(CHANNELS)])
        act_ind = np.argmax(self.Q_sa(net_input)) if p > AC_MAX else p
        self.D[t%REPLAY_MEMORY, self.board_size_flat] = act_ind # a_t
        return self.ACTIONS[act_ind]

    def get_state(self):
        # self.model.save('my_model.h5')
        return 'my_model.h5'

    def Q_sa(self, st):
        st = st.reshape(BATCH, CHANNELS, self.board_size[0], self.board_size[1])
        ret = self.model.predict(st, batch_size=1, verbose=1)
        return np.argmax(ret)

    def buildmodel(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), init='he_normal', border_mode='same',
            input_shape=(CHANNELS, self.board_size_flat)))
        model.add(Reshape(CHANNELS, self.board_size[0], self.board_size[1]))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), init='he_normal', border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal', border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3, init='he_normal'))

        adam = Adam(lr=1e-3)
        model.compile(loss='mse', optimizer=adam)
        print("Model built")
        return model

