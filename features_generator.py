import matplotlib.pyplot as plt
import numpy as np
import episode_parser
import scipy.ndimage as ndim


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
    for i in range(1, 10):
        apple_num = -i
        apple_map = board == apple_num
        apple_dists = apple_map * dist_map
        apple_dists[apple_dists==0] = apple_dists.max()
        ind = np.unravel_index(np.argmin(apple_dists, axis=None), apple_dists.shape)
        a=1


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
    episode_to_features_vec(episodeArray[777])