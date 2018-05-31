from policies import base_policy as bp
import numpy as np
import pickle
from keras.models import model_from_json
import  episode_parser
import  features_generator
import queue
from keras import optimizers
from keras.models import Sequential

class PolicyRafaelTest(bp.Policy):

    def cast_string_args(self, policy_args):
        self.model_path = policy_args["model_path"]
        return policy_args

    def init_run(self):
        try:
            # load json and create model
            json_file = open(self.model_path + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights(self.model_path + '.h5')
            print("Loaded model from disk")
            adam = optimizers.Adam(lr=0.001)  # , decay=0.01)
            self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])

            self.epsilon = 0.1 ## noexploration as we test a pure policy
            self.r_sum = 0
            #statistics:
            self.loss = []

        except IOError:
            state = np.zeros(100)

    def learn(self, reward, t):
        self.r_sum += reward
        self.log(str(self.r_sum), 'r_sum')

    def act(self, t, state, player_state):
        rand_num = np.random.rand()
        if (rand_num<self.epsilon):
            rand_action_index = np.random.randint(3)
            print("random action: " + bp.Policy.ACTIONS[rand_action_index])
            self.epsilon *= 0.9999
            selected_action = bp.Policy.ACTIONS[rand_action_index]
        else:
            scores = np.zeros((3, 1))
            i=0
            for next_dir in bp.Policy.ACTIONS:
                current_episode = episode_parser.parseEpisodeFromState(player_state,state,t, next_dir)
                features = features_generator.episode_to_features_vec(current_episode)
                features = np.reshape(features,(1,features.shape[0]))
                scores[i] = self.model.predict(features)[0][0]
                i += 1
            max_score = np.argmax(scores)
            selected_action = bp.Policy.ACTIONS[max_score]
            print("selected action: "+selected_action)

        return selected_action

    def get_state(self):
        return None


