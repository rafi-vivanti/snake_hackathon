import matplotlib.pyplot as plt
import  numpy as np


def show_graph(filename):
    loss = []
    reward = []
    max_iter = 20000000
    with open(filename) as f:
        for i,line in enumerate(f):
            nums = (line.split(' '))
            reward.append(float(nums[0]))
            if i > max_iter:
                break

    clear_reward = []
    games_duration = []
    prev = 0
    cntr = 0
    for i in reward:
        if i==0:
            clear_reward.append(prev)
            games_duration.append(cntr)
            cntr = 0
        cntr += 1

        prev = i
    # clear_reward1 = clear_reward[:5686]
    clear_reward2 = clear_reward #[5686:]
    # plt.plot(clear_reward1,'.')
    plt.plot(clear_reward2,'.')
    plt.hold(1)
    filter_length = 100.
    temp = np.ones(int(filter_length))/filter_length
    # clear_reward1_smooth = np.correlate(clear_reward1, temp)
    clear_reward2_smooth = np.correlate(clear_reward2, temp)
    games_duration_smooth = np.correlate(np.asarray(games_duration)*6., temp)
    # plt.plot(clear_reward1_smooth)
    plt.plot(clear_reward2_smooth)
    plt.plot(games_duration_smooth)
    plt.show()



if __name__ == '__main__':

    # filename = r"D:\projects\RL\snake\hackathon\rafi\models\loss\losses3.txt"
    # filename = r"D:\projects\RL\snake\hackathon\rafi\models\loss\policy_rafael.txt"
    filename = r"D:\projects\RL\snake\hackathon\rafi\models\loss\policy_avoid_collision.txt"
    show_graph(filename)