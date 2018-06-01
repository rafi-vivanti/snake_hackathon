from policies import base_policy as bp
import numpy as np
import pickle
from keras.models import model_from_json
import  episode_parser
import  features_generator
import queue
from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras import initializers


class PolicyRafael(bp.Policy):

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

            # self.model.load_weights(self.model_path + '.h5')
            self.reset_weights()
            print("Loaded model from disk")
            adam = optimizers.Adam(lr=0.001)  # , decay=0.01)
            self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])

            self.epsilon = 0.5
            self.MAX_EXPERIENCE = 500
            self.BATCH_SIZE = 100
            self.learning_q_rewards = queue.Queue(3) #3 is the max size
            self.learning_q_features = queue.Queue(3)  # 3 is the max size
            self.experience_table= []

            #statistics:
            self.loss = []
            self.r_sum = 0

            #state = pickle.load(open(self.load_from))
        except IOError:
            state = np.zeros(100)
        #self.state = state


    def learn(self, reward, t):
        self.r_sum += reward
        if reward == -100:
            self.r_sum = 0
        gamma = 0.2
        if (self.learning_q_rewards.full()):
            #calculate rewrad_sum for 3-previous state
            reward_sum = 0
            cur_pow = 0
            for i in self.learning_q_rewards.queue:
                reward_sum += pow(gamma,cur_pow)*i
                cur_pow += 1
                if i == -100:
                    gamma = 0 ## we are in the next game already, calculating reward for last episodes of previous game.
            reward_sum += pow(gamma, cur_pow) * reward
            cur_pow += 1
            learning_features = np.reshape(self.learning_q_features.queue[0], (1, self.learning_q_features.queue[0].shape[0]))
            reward_sum += pow(gamma, cur_pow) * self.model.predict(learning_features)[0][0]

            self.experience_table.append(np.hstack((self.learning_q_features.get(), reward_sum)))
            if len(self.experience_table) > self.MAX_EXPERIENCE:
                self.experience_table = self.experience_table[1:]
            self.learning_q_rewards.get()
        self.learning_q_rewards.put(reward)
        self.train_net(t)


    def train_net(self, iter_num):
        if len(self.experience_table)>self.BATCH_SIZE:
            # print("train net:")
            sample_ind = np.random.choice(len(self.experience_table),self.BATCH_SIZE)
            samples = [self.experience_table[i] for i in sample_ind ]
            X = np.asarray(samples)[:,:-1]
            y = np.asarray(samples)[:, -1]
            hist = self.model.fit(X, y, verbose=0)
            loss = hist.history['loss'][0]
            mae = hist.history['mean_absolute_error'][0]
            self.loss.append([loss, mae, self.r_sum])

            if np.mod(iter_num, 10000) == 0:
                # serialize model to JSON
                model_json = self.model.to_json()
                import time
                t = time.clock()

                name = r'D:\projects\RL\snake\hackathon\rafi\models\saved_models3\model_' + str(t)
                with open(name + '.json', "w") as json_file:
                    json_file.write(model_json)
                self.model.save_weights(name + '.h5')
                print("Saved model to disk")
                f = open(r'D:\projects\RL\snake\hackathon\rafi\models\loss\losses3.txt' , 'a')
                for i in range(len(self.loss)):
                    f.write(str(self.loss[i][0]) + " ")
                    f.write(str(self.loss[i][1]) + " ")
                    f.write(str(self.loss[i][2]))
                    f.write('\n')
                self.loss = []
                f.close()

    def act(self, t, state, player_state):
        rand_num = np.random.rand()
        if (rand_num<self.epsilon):
            rand_action_index = np.random.randint(3)
            # print("random action: " + bp.Policy.ACTIONS[rand_action_index])
            self.epsilon *= 0.99999
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
            # print("selected action: "+selected_action)

        selcted_episode = episode_parser.parseEpisodeFromState(player_state, state, t, selected_action)
        features = features_generator.episode_to_features_vec(selcted_episode)
        self.learning_q_features.put(features)

        print("epsilon: " + str(self.epsilon) + "/n")
        return selected_action

    def get_state(self):
        return None

    def reset_weights(self):
        session = K.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

    def reset_model(self):
        for layer in self.model.layers:
            if hasattr(layer, 'init'):
                init = getattr(layer, 'init')
                new_weights = init(layer.get_weights()[0].shape).get_value()
                bias = shared_zeros(layer.get_weights()[1].shape).get_value()
                layer.set_weights([new_weights, bias])
