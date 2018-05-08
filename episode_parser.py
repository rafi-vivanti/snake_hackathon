import numpy as np
import re


class Episode:
    def __init__(self):
        self.reward = 0
        self.r_sum = 0
        self.cycle = 0
        self.board = []
        self.snake_head = []
        self.dir = 'W'
        self.next_dir = 'CW'

def parseBoard(lines):
    board = np.zeros((0,10), dtype=int)
    for line in lines:
        line = line.replace(']', '')
        line = line.replace('[', '')
        line = line.strip()
        tokens = line.split()
        arr = np.asarray(tokens).astype(int)
        board = np.vstack((board, arr))
    return board


def parseHead(line):
    line = line.replace('(', '')
    line = line.replace(')', '')
    line = line.replace(',', ' ')
    line = line.strip()
    tokens = line.split()
    arr = np.asarray(tokens).astype(int)
    return arr


def parse_episode(lines):
    episode_instance = Episode()
    episode_instance.cycle = int(lines[2].split()[2])
    # print (episode.cycle)

    episode_instance.reward = int(lines[0].split()[3])
    episode_instance.r_sum = int(lines[1].split()[3])
    episode_instance.dir = lines[13][-2]
    head_line = lines[13]
    episode_instance.snake_head = parseHead(head_line[:-2])
    episode_instance.board = parseBoard(lines[3:13])# np.asarray(int(lines[3:13, :-1]))
    episode_instance.next_dir = lines[14].split()[-1]
    return episode_instance


def parse(file_name):
    f = open(file_name)
    content = f.readlines()
    content = content[1:]
    i = 0
    episodeArray = []
    while (i<len(content)):
        lines=content[i:i+15]
        episode_instance = parse_episode(lines)
        episodeArray.append(episode_instance)
        i+=15
    return episodeArray


if __name__ == '__main__':
    file_name = r"D:\projects\RL\snake\hackathon\rafi\new_logs\log_0.txt"
    episodeArray = parse(file_name)
    a=1