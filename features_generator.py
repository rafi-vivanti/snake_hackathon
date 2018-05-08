import matplotlib.pyplot as plt
import numpy as np
import episode_parser
import scipy.ndimage as ndim
import os

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

def get_data_for_the_next_steps(episode):
    sz = episode.board.shape
    head = episode.snake_head
    real_head = np.asarray([np.mod(head[0], sz[0]), np.mod(head[1], sz[1])])

    board_copy=episode.board.copy()
    board_copy[real_head[0],real_head[1]]=777


    if episode.dir == 'S':
        north_board = np.rot90(board_copy, 2);
    elif episode.dir == 'W':
        north_board = np.rot90(board_copy, 3);
    elif episode.dir == 'E':
        north_board = np.rot90(board_copy, 1);
    else:
        north_board = board_copy.copy()


    new_head_position = np.unravel_index(np.argmax(north_board, axis=None), north_board.shape)

    north_board[north_board==777]=1
    big_north_board = np.hstack((north_board, north_board, north_board))
    big_north_board = np.vstack((big_north_board, big_north_board, big_north_board))
    sub_board = big_north_board[new_head_position[0]+sz[0]-3:new_head_position[0]+sz[0]+4, new_head_position[1]+sz[1]-3:new_head_position[1]+sz[1]+4]

    if not sub_board.shape == (7,7):
        print("Error: sub_board.shape != (7,7)")

    mid_col = int(np.floor(sub_board.shape[0] / 2))
    mid_row = int(np.floor(sub_board.shape[1] / 2))
    values = np.zeros_like(sub_board)
    struct_ = ndim.generate_binary_structure(2, 1)
    values[mid_row, mid_col] = 1
    values = ndim.binary_dilation(values, structure=struct_).astype(values.dtype)
    values = ndim.binary_dilation(values, structure=struct_).astype(values.dtype)
    values = ndim.binary_dilation(values, structure=struct_).astype(values.dtype)
    sub_board = sub_board[values > 0]
    next_direction_code = 0
    if episode.next_dir == 'CW':
        next_direction_code = 1
    elif episode.next_dir == 'CC':
        next_direction_code = -1

    return np.hstack((next_direction_code, sub_board))



def episode_to_features_vec(episode):
    sz = episode.board.shape
    head = episode.snake_head
    board = episode.board
    real_head = np.asarray([np.mod(head[0], sz[0]), np.mod(head[1], sz[1])])

    feature_vector = get_data_for_the_next_steps(episode)

    dist_map = get_distance_map(board, real_head)

    for i in range(1, 10): # we only support 10 types of apples

        apple_num = -i
        apple_map = board == apple_num
        if not apple_map.any():
            feature_vector = np.hstack((feature_vector,board.nbytes/8)) # should be very large number
            feature_vector = np.hstack((feature_vector,0))
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
                step = 1
            else:
                step = 0

        feature_vector =np.hstack((feature_vector,np.linalg.norm(diff_index)))
        feature_vector = np.hstack((feature_vector,step))
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


def all_logs_to_features(folder):
    big_res_matix= np.zeros((0, 45))
    i=0
    for file_name in os.listdir(folder):
        print(i)
        i=i+1
        file_path= os.path.join(folder,file_name)
        episodeArray = episode_parser.parse(file_path)
        fetures_table = np.asarray(episode_to_features_table(episodeArray))
        rewards_vec = np.asarray(episodes_to_rewards_vec(episodeArray)).transpose()
        rewards_vec = np.resize(rewards_vec, (rewards_vec.shape[0], 1))
        one_log_res = np.hstack((fetures_table,rewards_vec))
        big_res_matix = np.vstack((big_res_matix,one_log_res))
    return big_res_matix


if __name__ == '__main__':
    folder = r"D:\projects\RL\snake\hackathon\rafi\logs"
    big_res_matix = all_logs_to_features(folder)
    np.save(os.path.join(folder,"feturesAndRewards"), big_res_matix)