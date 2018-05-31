import matplotlib.pyplot as plt
import  numpy as np


def show_graph(filename):
    f = open(filename)
    lines = f.readlines()
    loss = []
    reward = []
    for line in lines:
        nums = (line.split(' '))
        loss.append(float(nums[0]))
        reward.append(float(nums[2]))

    clear_reward = []
    prev = 0
    for i in reward:
        if i==0:
            clear_reward.append(prev)
        prev = i
    # clear_reward1 = clear_reward[:5686]
    clear_reward2 = clear_reward[5686:]
    # plt.plot(clear_reward1,'.')
    plt.plot(clear_reward2,'.')
    plt.hold(1)
    temp = np.ones(1000)/1000.
    # clear_reward1_smooth = np.correlate(clear_reward1, temp)
    clear_reward2_smooth = np.correlate(clear_reward2, temp)
    # plt.plot(clear_reward1_smooth)
    plt.plot(clear_reward2_smooth)
    plt.show()



if __name__ == '__main__':

    filename = r"D:\projects\RL\snake\hackathon\rafi\models\loss\losses2.txt"
    show_graph(filename)