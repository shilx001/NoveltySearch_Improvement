import numpy as np
from GA import *
import gym
import tensorflow as tf
import datetime
from ant_maze_env import *


class HP:
    def __init__(self, env, seed=1, input_dim=None, output_dim=None, hidden_size=64):
        self.env = env
        self.input_dim = self.env.observation_space.shape[0] if input_dim is None else input_dim
        self.output_dim = self.env.action_space.shape[0] if output_dim is None else output_dim
        self.seed = seed
        self.hidden_size = hidden_size
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.env.seed(seed)


class Policy:
    def __init__(self, hp, params=None):
        # 根据parameter初始化
        self.hp = hp
        self.input_dim, self.output_dim, self.hidden_size = hp.input_dim, hp.output_dim, hp.hidden_size
        self.param_count = hp.input_dim * hp.hidden_size + 2 * hp.hidden_size + hp.hidden_size * hp.hidden_size + hp.hidden_size * (
            1 + hp.output_dim)
        if params is not None:
            assert len(params) == self.param_count
        self.params = params

    def get_params_count(self):
        return self.param_count

    def evaluate_tf(self, state):
        # 得出action值，三层神经网络？用parameter来说
        sess = tf.Session()
        input_state = np.reshape(state, [1, self.input_dim])
        feed_state = tf.placeholder(dtype=tf.float64, shape=[1, self.input_dim])
        param_list = self.get_detail_params()
        l1 = tf.nn.relu(tf.matmul(feed_state, param_list[0]) + param_list[1])
        l2 = tf.nn.relu(tf.matmul(l1, param_list[2])) + param_list[3]
        output_action = tf.nn.tanh(tf.matmul(l2, param_list[4]) + param_list[5])
        return sess.run(output_action, feed_dict={feed_state: input_state})

    def evaluate(self, state):
        input_state = np.reshape(state, [1, self.input_dim])
        param_list = self.get_detail_params()
        l1 = np.maximum(0, input_state.dot(param_list[0]) + param_list[1])
        l2 = np.maximum(0, l1.dot(param_list[2]) + param_list[3])
        return np.tanh(l2.dot(param_list[4]) + param_list[5])

    def get_detail_params(self):
        # 得到w1,b1,w2,b2,w3,b3
        w1 = self.params[:self.input_dim * self.hidden_size]
        b1 = self.params[len(w1):len(w1) + self.hidden_size]
        w2 = self.params[len(w1) + len(b1):len(w1) + len(b1) + self.hidden_size * self.hidden_size]
        b2 = self.params[len(w1) + len(b1) + len(w2):len(w1) + len(b1) + len(w2) + self.hidden_size]
        w3 = self.params[len(w1) + len(w2) + len(b1) + len(b2):len(w1) + len(w2) + len(b1) + len(
            b2) + self.hidden_size * self.output_dim]
        b3 = self.params[-self.output_dim:]
        return [np.reshape(w1, [self.input_dim, self.hidden_size]),
                np.reshape(b1, [self.hidden_size, ]),
                np.reshape(w2, [self.hidden_size, self.hidden_size]),
                np.reshape(b2, [self.hidden_size, ]),
                np.reshape(w3, [self.hidden_size, self.output_dim]),
                np.reshape(b3, [self.output_dim, ])]

    def get_fitness(self):
        total_reward = 0
        target_goal = np.array([0, 16])
        env = self.hp.env
        obs = env.reset()
        for step in range(1000):
            #env.render()
            action = self.evaluate(obs)
            next_obs, reward, done, _ = env.step(action)
            if np.sum(np.square(next_obs[:2]-target_goal))<0.1:
                reward = 1000
            else:
                reward = -1
            obs = next_obs
            total_reward += reward
            if done or step is 999:
                break
        return total_reward


env = AntMazeEnv(maze_id='Maze')

hp = HP(env=env, input_dim=30, output_dim=8)
policy = Policy(hp)
ga = GA(num_params=policy.get_params_count(), pop_size=10, elite_frac=0.2)

for episode in range(1000):
    start_time = datetime.datetime.now()
    population = ga.ask()
    fitness = [Policy(hp, params=_).get_fitness() for _ in population]
    ga.tell(population, fitness)
    print('######')
    print('Episode ', episode)
    print('Best fitness value: ', max(fitness))
    print('Running time:', (datetime.datetime.now() - start_time).seconds)
