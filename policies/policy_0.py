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

    def learn(self, reward, t):
        #if t % 1 == 0:
        self.log(str(reward), 'reward')

           # self.r_sum = 0
       # else:
        self.r_sum += reward
        self.log(str(self.r_sum), 'r_sum')

    def act(self, t, state, player_state):
        # print ("act")
        head_pos = player_state['chain'][-1]
        self.log(str(t) + "\n" + str(state) + "\n" + str(head_pos) + player_state['dir'])
        a = bp.Policy.ACTIONS[min(np.random.randint(20), 2)]  # 10% of actions are random
        for a in [a] + list(np.random.permutation(bp.Policy.ACTIONS)):
            r, c = head_pos.move(bp.Policy.TURNS[player_state['dir']][a]) % state.shape
            if state[r, c] <= 0: return a
        return a

    def get_state(self):
        return None