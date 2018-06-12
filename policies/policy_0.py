from policies import base_policy as bp
import numpy as np


class AvoidCollisions(bp.Policy):

    def cast_string_args(self, policy_args):
        policy_args['example'] = int(policy_args['example']) if 'example' in policy_args else 0
        return policy_args

    def init_run(self):
        #print(self.example)
        self.r_sum = 0
        self.log("hello log")
        self.reward=[]
       # self.loss = []

    def learn(self, reward, t):
        #if t % 1 == 0:
        self.log(str(reward), 'reward')

           # self.r_sum = 0
       # else:
        self.r_sum += reward
        if reward == -100:
            self.r_sum = 0
        # self.log(str(self.r_sum), 'r_sum')
        self.reward.append(self.r_sum)
        if np.mod(t, 10000) == 0:

            f = open(r'D:\projects\RL\snake\hackathon\rafi\models\loss\policy_avoid_collision.txt', 'a')
            for i in range(len(self.reward)):
                f.write(str(self.reward[i]))  ## only current reward
                f.write('\n')
            #self.loss = []
            f.close()

    def act(self, t, state, player_state):
        # print ("act")
        head_pos = player_state['chain'][-1]
        self.log(str(t) + "\n" + str(state) + "\n" + str(head_pos) + player_state['dir'])
        # a = bp.Policy.ACTIONS[min(np.random.randint(20), 2)]  # 10% of actions are random
        rand_int = min(np.random.randint(6), 2)# 33% of actions are random
        # print(rand_int)
        a = bp.Policy.ACTIONS[rand_int]
        action = a
        # print(a)
        if rand_int < 2:
            # print ('random action ' + str(t))
            action = a
            # print(action)

        else:
            # print('regular action ' + str(t))
            for a in [a] + list(np.random.permutation(bp.Policy.ACTIONS)):
                r, c = head_pos.move(bp.Policy.TURNS[player_state['dir']][a]) % state.shape
                if state[r, c] <= 0:
                    action = a
                    # print(action)
                    break
                action = a
        self.log(action, 'next action')
        return action

    def get_state(self):
        return None
