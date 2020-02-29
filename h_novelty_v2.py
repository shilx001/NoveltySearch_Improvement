import numpy as np
from GA import *
import gym
import tensorflow as tf
import datetime
from ant_maze_env import *
from algorithms.td3_network import *
import pickle

#linear meta control policy
class HP:
    def __init__(self, env, seed=1, input_dim=30, output_dim=2, action_bound=20, hidden_size=2, pop_size=200):
        self.env = env
        self.input_dim = self.env.observation_space.shape[0] if input_dim is None else input_dim
        self.output_dim = self.env.action_space.shape[0] if output_dim is None else output_dim
        self.seed = seed
        self.pop_size = pop_size
        self.action_bound = action_bound
        self.hidden_size = hidden_size
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.env.seed(seed)
        self.archive = []

    def cal_novelty(self, position):
        if len(self.archive) < 10:
            return 0
        all_data = np.reshape(self.archive, [-1, 2])
        p = np.reshape(position, [-1, 2])
        dist = np.sqrt(np.sum(np.square(p - all_data), axis=1))
        return np.mean(np.sort(dist)[:15])


class Policy:
    def __init__(self, hp, params=None):
        # 根据parameter初始化
        self.hp = hp
        self.input_dim, self.output_dim, self.hidden_size = hp.input_dim, hp.output_dim, hp.hidden_size
        self.param_count = hp.input_dim * hp.hidden_size + self.hidden_size
        if params is not None:
            assert len(params) == self.param_count
        self.params = params

    def get_params_count(self):
        return self.param_count

    def set_params(self, params):
        self.params = params

    def evaluate(self, state):
        input_state = np.reshape(state, [1, self.input_dim])
        param_list = self.get_detail_params()
        l1 = np.maximum(0, input_state.dot(param_list[0]) + param_list[1])
        return np.tanh(l1) * self.hp.action_bound

    def get_detail_params(self):
        # 得到w1,b1,w2,b2,w3,b3
        w1 = self.params[:self.input_dim * self.hidden_size]
        b1 = self.params[len(w1):len(w1) + self.hidden_size]

        return [np.reshape(w1, [self.input_dim, self.hidden_size]),
                np.reshape(b1, [self.hidden_size, ])]

    def get_fitness(self):  # 这个没用了，不需要
        total_reward = 0
        target_goal = np.array([0, 16])
        env = self.hp.env
        obs = env.reset()
        for step in range(1000):
            # env.render()
            action = self.evaluate(obs)
            next_obs, reward, done, _ = env.step(action)
            if np.sum(np.square(next_obs[:2] - target_goal)) < 0.1:
                reward = 1000
            else:
                reward = -1
            obs = next_obs
            total_reward += reward
            if done or step is 999:
                break
        return total_reward


class MetaPolicy:
    def __init__(self, hp):
        self.hp = hp
        self.h_policy = Policy(hp)  # hp只是higher policy的params!
        self.l_policy = TD3(state_dim=32, action_dim=8, action_bound=30, hidden_size=64)
        self.optimizer = GA(num_params=self.h_policy.get_params_count(), pop_size=hp.pop_size)

    def get_fitness(self, params):
        # 根据h_policy的params得到fitness, for 1 rollout.
        self.h_policy.set_params(params)
        env = self.hp.env
        obs = env.reset()
        h_reward = 0
        target_goal = np.array([0, 16])
        for step in range(500):
            if step % 10 == 0:  # 如果为0则set goal
                current_goal = self.h_policy.evaluate(obs).flatten()
            state_goal = np.hstack((obs.flatten(), current_goal))
            action = self.l_policy.get_action(state_goal)  # evaluate primitive action
            next_obs, reward, done, _ = env.step(action)
            l_reward = -1 * (np.sqrt(np.sum(np.square(obs[:2] + current_goal - next_obs[:2]))) - 0.05 > 0)
            l_done = l_reward == 0
            next_state_goal = np.hstack((next_obs.flatten(), current_goal))
            self.l_policy.store(state_goal, next_state_goal, action, l_reward, l_done)
            if np.sqrt(np.sum(np.square(next_obs[:2] - target_goal))) < 0.1:
                reward = 0
            else:
                reward = -1
            h_reward += reward
            obs = next_obs
            if done:
                break
        self.l_policy.train(5)
        final_pos = obs[:2]
        return h_reward, final_pos

    def train(self, num_generation):
        fitness_list = []
        final_pos_list = []
        for episode in range(num_generation):
            start_time = datetime.datetime.now()
            population = self.optimizer.ask()
            fitness = []
            final_pos = []
            novelty = []
            for i in range(self.hp.pop_size):
                f, p = self.get_fitness(params=population[i])
                fitness.append(f)
                final_pos.append(p)
                novelty.append(self.hp.cal_novelty(p))
            for i in final_pos:  # 加入archive
                self.hp.archive.append(i)
            self.optimizer.tell(population, novelty)
            best_idx = np.argmax(novelty)
            fitness_list.append(max(fitness))
            final_pos_list.append(final_pos[best_idx])
            print('######')
            print('Episode ', episode)
            print('Best fitness value: ', max(fitness))
            print('Best novelty: ', novelty[best_idx])
            print('Final distance:', np.sqrt(np.sum(np.square(final_pos[best_idx] - np.array([0, 16])))))
            print('Running time:', (datetime.datetime.now() - start_time).seconds)
        return fitness_list, final_pos_list


env = AntMazeEnv(maze_id='Maze')
hp = HP(env=env)
meta_policy = MetaPolicy(hp)
fitness, final_pos = meta_policy.train(1000)
pickle.dump(fitness, open('hga_novelty_v2_fitness', mode='wb'))
pickle.dump(final_pos, open('hga_novelty_v2_final_pos', mode='wb'))
