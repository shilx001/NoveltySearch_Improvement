import numpy as np
import gym
import tensorflow as tf
import datetime
from ant_maze_env import *
import pickle
import autograd.numpy as anp
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize

# multi-objective evolution algorithm

TASK_NAME = 'Maze'
VERSION = '1'
POPULATION = 50
EPISODE_NUMBER = 1000

if TASK_NAME is 'Maze':
    TARGET_GOAL = np.array([0, 16])
elif TASK_NAME is 'Push':
    TARGET_GOAL = np.array([0, 19])
elif TASK_NAME is 'Fall':
    TARGET_GOAL = np.array([0, 27])


# run the GA with novelty search
# 2020-2-8
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
        self.archive = []  # store the final position of the states
        self.count = 0  # calculate the number of episodes
        self.current_bc_list = []
        self.bc_list = []

    def cal_novelty(self, position):
        #
        all_data = np.reshape(self.archive, [-1, 2])
        p = np.reshape(position, [-1, 2])
        dist = np.sqrt(np.sum(np.square(p - all_data), axis=1))
        return np.mean(np.sort(dist)[:15])

    def get_update_signal(self):
        if self.count % 50 is 0:
            return True
        else:
            return False


class Policy:
    def __init__(self, hp, params=None):
        # 根据parameter初始化
        self.hp = hp
        self.input_dim, self.output_dim, self.hidden_size = hp.input_dim, hp.output_dim, hp.hidden_size
        self.param_count = hp.input_dim * hp.hidden_size + self.hidden_size + self.hidden_size * self.hidden_size + \
                           self.hidden_size + self.hidden_size * self.output_dim + self.output_dim
        if params is not None:
            print(self.param_count)
            assert len(params) == self.param_count
        self.params = params

    def set_params(self, params):
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
        output_action = tf.nn.tanh(tf.matmul(l2, param_list[4]) + param_list[5]) * 30
        return sess.run(output_action, feed_dict={feed_state: input_state})

    def evaluate(self, state):
        input_state = np.reshape(state, [1, self.input_dim])
        param_list = self.get_detail_params()
        l1 = np.maximum(0, input_state.dot(param_list[0]) + param_list[1])
        l2 = np.maximum(0, l1.dot(param_list[2]) + param_list[3])
        return 30 * np.tanh(l2.dot(param_list[4]) + param_list[5])

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
        env = self.hp.env
        obs = env.reset()
        for step in range(500):
            # env.render()
            action = self.evaluate(obs)
            next_obs, reward, done, _ = env.step(action)
            if np.sqrt(np.sum(np.square(next_obs[:2] - TARGET_GOAL))) < 0.1:
                reward = 1
            else:
                reward = 0
            obs = next_obs
            total_reward += reward
            if done:
                break
        if len(self.hp.archive) > 0:
            novelty = self.hp.cal_novelty(obs[:2])
        else:
            novelty = 0
        self.hp.count += 1
        return -novelty, -total_reward,obs[:2]


class MyProblem(Problem):
    def __init__(self, n_var, hp):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0,
                         xl=anp.array(-10000 * np.ones([n_var, ])),
                         xu=anp.array(10000 * np.ones([n_var, ])))
        self.hp = hp

    def _evaluate(self, x, out, *args, **kwargs):
        novelty = []
        fitness = []
        for x1 in x:
            policy = Policy(hp=self.hp, params=x1)
            n, f, b = policy.get_fitness()
            novelty.append(n)
            fitness.append(f)
            self.hp.current_bc_list.append(b)
        out["F"] = anp.column_stack([fitness, novelty])
        # out["G"] = anp.column_stack(np.zeros([50,2]))
        if self.hp.get_update_signal():
            # 存储每一个当前bc list中的bc进novelty
            best_idx = np.argmin(novelty)
            print('########')
            print('Generation:',int(self.hp.count/50))
            print('Max novelty is:', np.min(novelty))
            print('Position:', self.hp.current_bc_list[best_idx])
            print('Archive length:', len(self.hp.archive))
            for bc in self.hp.current_bc_list:
                self.hp.archive.append(bc)
            self.hp.current_bc_list = []


env = AntMazeEnv(maze_id=TASK_NAME)
hp = HP(env=env, input_dim=30, output_dim=8)
policy = Policy(hp)
algorithm = NSGA2(pop_size=50, n_offsprings=50, sampling=get_sampling("real_random"),
                  crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                  mutation=get_mutation("real_pm", eta=20),
                  eliminate_duplicates=True)
termination = get_termination("n_gen", 1000)
problem = MyProblem(n_var=policy.get_params_count(), hp=hp)
res = minimize(problem, algorithm, termination, seed=1)
pickle.dump(problem.hp.archive,open(TASK_NAME+'_moea_'+str(VERSION),mode='wb'))
