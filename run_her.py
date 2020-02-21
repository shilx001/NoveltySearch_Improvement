# hindsight experience replay

import numpy as np
from GA import *
import gym
import datetime
from ant_maze_env import *
from algorithms.ddpg import *


def cal_reward(goal, current_state):
    return -int(np.sqrt(np.sum((goal - current_state[:2]) ** 2)) > 0.1)


goal = np.array([0, 16])

agent = DDPG(state_dim=30 + 2, action_dim=8, action_bound=30)
for episode in range(5000):
    env = AntMazeEnv(maze_id='Maze')
    obs = env.reset()
    current_path = []
    for step in range(500):  # normal experience replay
        state_goal = np.hstack([obs, goal])
        action = agent.get_action(state_goal) + np.random.randn(1, 8) * 10
        next_obs, reward, done, _ = env.step(action)
        reward = cal_reward(goal, next_obs)  # 普通experience replay
        if step is 499 or reward is 0:
            done = True
        next_obs_goal = np.hstack([next_obs, goal])
        current_path.append([obs, action, next_obs])
        agent.store(state_goal, next_obs_goal, action, reward, done)
        if done:
            break
    # 针对该trajectory随机选择3个goal进行hindsight experience replay, 随机选择final goal 不太好，应该选择最后的就好。
    # goal_index = np.random.choice(len(current_path), 3, replace=False)
    goal_index = [-1]
    for i in goal_index:  # 针对每个goal
        temp_element = current_path[i]
        temp_state = temp_element[-1]
        current_goal = temp_state[:2]
        for c_state, c_action, c_nstate in current_path:
            c_reward = cal_reward(current_goal, c_nstate)
            if c_reward is 0:
                c_done = True
            else:
                c_done = False
            agent.store(np.hstack([current_goal, c_state]), np.hstack([current_goal, c_nstate]), c_action, c_reward,
                        c_done)
    agent.train(100)  # learning step
    print('Episode ', episode, 'get goal:', done, 'at step ', step)
