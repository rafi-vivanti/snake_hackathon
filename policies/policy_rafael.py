from policies import base_policy as bp
import numpy as np
import pickle
from keras.models import model_from_json
import  episode_parser
import  features_generator
import queue
from keras.models import Sequential

class PolicyRafael(bp.Policy):

    def cast_string_args(self, policy_args):
        return policy_args

    def init_run(self):
        try:
            model_path = r'D:\projects\RL\snake\hackathon\rafi\models\first_model.h5'

            # load json and create model
            json_file = open(r'D:\projects\RL\snake\hackathon\rafi\models\first_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights(r'D:\projects\RL\snake\hackathon\rafi\models\first_model.h5')
            print("Loaded model from disk")
            self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            self.epsilon = 0.1
            # self.step = 0
            # self.prev_0_reward=0
            # self.prev_1_reward=0
            # self.prev_2_reward=0
            # self.experience_table = []
            self.MAX_EXPERIENCE = 500
            self.BATCH_SIZE = 20
            self.learning_q_rewards = queue.Queue(3) #3 is the max size
            self.learning_q_features = queue.Queue(3)  # 3 is the max size
            self.experience_table= []

            #state = pickle.load(open(self.load_from))
        except IOError:
            state = np.zeros(100)
        #self.state = state
    def learn(self, reward, t):
        gamma = 0.2
        if (self.learning_q_rewards.full()):
            #calculate rewrad_sum for 3-previous state
            cur_pow = self.learning_q_rewards.maxsize
            reward_sum = 0
            cur_pow = 0
            for i in self.learning_q_rewards.queue:
                reward_sum += pow(gamma,cur_pow)*i
                cur_pow += 1
            reward_sum += pow(gamma, cur_pow) * reward
            self.experience_table.append(np.hstack((self.learning_q_features.get(),reward_sum)))
            if len(self.experience_table) > self.MAX_EXPERIENCE:
                self.experience_table = self.experience_table[1:]
            self.learning_q_rewards.get()
        self.learning_q_rewards.put(reward)
        self.train_net()
    # def learn(self, reward, t):
    #     gamma = 0.2
    #     self.step += 1
    #     if (self.q.qsize() < 4):
    #         if (len(self.experience_table))>0:
    #             self.experience_table=self.experience_table[:-1]
    #     else:
    #         if (self.q.qsize()>4):
    #             self.q.get()
    #         reward_sum = reward
    #         reward_sum += gamma * self.prev_0_reward
    #         reward_sum += gamma *gamma * self.prev_1_reward
    #         reward_sum += gamma *gamma *gamma * self.prev_2_reward
    #         features_to_predict = np.reshape(self.experience_table[-1][:-1], (1, self.experience_table[-1][:-1].shape[0]))
    #         reward_sum += gamma *gamma *gamma *gamma *self.model.predict(features_to_predict)
    #
    #         if (reward == -100):
    #             self.step = 0
    #             self.prev_0_reward = 0
    #             self.prev_1_reward = 0
    #             self.prev_2_reward = 0
    #         self.experience_table[-1][-1] = reward_sum
    #     self.train_net();


    def train_net(self):
        if len(self.experience_table)>self.BATCH_SIZE:
            sample_ind = np.random.choice(len(self.experience_table),self.BATCH_SIZE)
            samples = [self.experience_table[i] for i in sample_ind ]
            print ("train net:")
            #print ("exp:\n "+self.experience_table)
            #TODO actually train

    # def get_batch(self):
    #     batch = np.zeros((BATCH, CHANNELS, self.board_size_flat))
    #     sample_ind = np.random.choice(range(self.MAX_EXPERIENCE), replace=False, size=BATCH)
    #     samples = self.D[sample_ind]
    #     for i, ii in enumerate(sample_ind):
    #         batch[ii] = np.vstack([self.D[i - j] for j in range(CHANNELS)])
    #     return batch


    def act(self, t, state, player_state):
        rand_num = np.random.rand()
        if (rand_num<self.epsilon):
            print("random action: ")
            rand_action_index = np.random.randint(3)
            print("random action: " + bp.Policy.ACTIONS[rand_action_index])
            self.epsilon*= 0.999
            selected_action =  bp.Policy.ACTIONS[rand_action_index]
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

        selcted_episode = episode_parser.parseEpisodeFromState(player_state, state, t, selected_action)
        features = features_generator.episode_to_features_vec(selcted_episode)
        features_for_exp = np.hstack((features.ravel(), 0))
        self.learning_q_features.put(features_for_exp)


        return selected_action

    def get_state(self):
        return None


