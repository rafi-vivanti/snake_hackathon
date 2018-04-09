import matplotlib.pyplot as plt
import numpy as np
import episode_parser
import scipy.ndimage as ndim


def episodes_to_rewards_vec(episodeArray):
    rewards = [i.reward for i in episodeArray]
    prev_index=-1
    lambda_ = 0.9
    rewards_sum=np.zeros_like(rewards)
    for i in range (len(rewards)):
        if rewards[i]==-100 or i==len(episodeArray)-1:
            rewards_sum[i]=rewards[i]
            for j in range (i-1,prev_index,-1):
                next_reward_sum = rewards_sum[j+1]
                cur_reward = rewards[j]
                cur_reward_sum = (1-lambda_)*cur_reward + lambda_* next_reward_sum ## more smooth reward, do not erward twice for long games
                # cur_reward_sum = cur_reward + lambda_* next_reward_sum ## original RL theory.
                rewards_sum[j] = cur_reward_sum
            prev_index = i
    return rewards_sum


def episode_to_features_table(episodeArray):
    feature_table = []
    for episode in episodeArray:
        feature_table.append(episode_to_features_vec(episode))
    return feature_table

def episode_to_features_vec(episode):
    sz = episode.board.shape
    head = episode.snake_head
    board = episode.board
    real_head = np.asarray([np.mod(head[0], sz[0]), np.mod(head[1], sz[1])])

    if episode.dir == 'S':
        left_place = [real_head[0] + 1, real_head[1]]
        right_place = [real_head[0] - 1, real_head[1]]
        front_place = [real_head[0], real_head[1] + 1]

    elif episode.dir == 'N':
        left_place = [real_head[0] - 1, real_head[1]]
        right_place = [real_head[0] + 1, real_head[1]]
        front_place = [real_head[0], real_head[1] - 1]
    elif episode.dir == 'W':
        left_place = [real_head[0], real_head[1] + 1]
        right_place = [real_head[0], real_head[1] - 1]
        front_place = [real_head[0] - 1, real_head[1]]
    else: # episode.dir == 'E':
        left_place = [real_head[0], real_head[1] - 1]
        right_place = [real_head[0], real_head[1] + 1]
        front_place = [real_head[0] + 1, real_head[1]]

    left_place = np.asarray([np.mod(left_place [0], sz[0]), np.mod(left_place [1], sz[1])])
    right_place= np.asarray([np.mod(right_place[0], sz[0]), np.mod(right_place[1], sz[1])])
    front_place= np.asarray([np.mod(front_place[0], sz[0]), np.mod(front_place[1], sz[1])])

    left_content = board[left_place[0], left_place[1]]
    right_content = board[right_place[0], right_place[1]]
    front_content = board[front_place[0], front_place[1]]

    feature_vector = [left_content, right_content, front_content]

    dist_map = get_distance_map(board, real_head)



    for i in range(1, 10): # we only support 10 types of apples

        apple_num = -i
        apple_map = board == apple_num
        if not apple_map.any():
            feature_vector.append(board.nbytes/8) # should be very large number
            feature_vector.append(0)
            continue
        apple_dists = apple_map * dist_map
        apple_dists[apple_dists==0] = apple_dists.max()+1 # to avoid get zero when minimize
        apple_dists[real_head[0],real_head[1]] = apple_dists.max()*2

        if episode.dir == 'S':
            apple_dists_copy = np.rot90(apple_dists, 2);
        elif episode.dir == 'W':
            apple_dists_copy = np.rot90(apple_dists, 3);
        elif episode.dir == 'E':
            apple_dists_copy = np.rot90(apple_dists, 1);
        else:
            apple_dists_copy = apple_dists.copy()

        new_head_position= np.unravel_index(np.argmax(apple_dists_copy, axis=None), apple_dists_copy.shape)
        closest_apple = np.unravel_index(np.argmin(apple_dists_copy, axis=None), apple_dists_copy.shape)
        diff_index= np.asarray(closest_apple) - np.asarray(new_head_position)

        # decide what is the direction to the closest apple. the board is already rotated to the north (snake view)
        # decide on direction solely on direction in x axis
        if (diff_index[1]>0):
            step = 1
        elif (diff_index[1]<0):
            step=-1
        else:
            if (diff_index[0]>0):
                step =1
            else:
                step=0
        feature_vector.append(np.linalg.norm(diff_index))
        feature_vector.append(step)
    return feature_vector




def get_distance_map(board, head):
    big = np.zeros((board.shape[0]*3, board.shape[1]*3))
    big[board.shape[0]+head[0], board.shape[1]+head[1]] = 1
    big_dist_map = ndim.morphology.distance_transform_edt(1-big)
    res = np.zeros_like(board)
    w = board.shape[0]
    h = board.shape[1]
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            res[x, y] = np.min((big_dist_map[x,y], big_dist_map[x+w,y], big_dist_map[x+2*w,y], big_dist_map[x,y+h], big_dist_map[x+w,y+h], big_dist_map[x+2*w,y+h], big_dist_map[x,y+2*h], big_dist_map[x+w,y+2*h], big_dist_map[x+2*w,y+2*h]))
    return  res


if __name__ == '__main__':
    file_name = r"D:\projects\RL\snake\hackathon\rafi\logs\log_0.txt"
    episodeArray = episode_parser.parse(file_name)
    fetures_table = episode_to_features_table(episodeArray)
    rewards_vec = episodes_to_rewards_vec(episodeArray)