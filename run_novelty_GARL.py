import numpy as np
from GA import *
import gym
import tensorflow as tf
import datetime
from ant_maze_env import *
import pickle
from algorithms.utils import ReplayBuffer
# from algorithms.ddpg import DDPG
from algorithms.ddpg_v2 import DDPG

TASK_NAME = 'Maze'
VERSION = '1'
POPULATION = 50
REPLACE = 10
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
    def __init__(self, env, seed=1, input_dim=None, output_dim=None, hidden_size=64, learning_step=100,
                 explore_range=0):
        self.env = env
        self.input_dim = self.env.observation_space.shape[0] if input_dim is None else input_dim
        self.output_dim = self.env.action_space.shape[0] if output_dim is None else output_dim
        self.seed = seed
        self.hidden_size = hidden_size
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.env.seed(seed)
        self.archive = []  # store the final position of the states
        self.replay_buffer = ReplayBuffer()
        self.learning_step = learning_step
        self.explore_range = explore_range
        self.ddpg_agent = DDPG(state_dim=30, action_dim=8, action_bound=30, actor_lr=1e-4, critic_lr=1e-4)

    def cal_novelty(self, position):
        #
        all_data = np.reshape(self.archive, [-1, 2])
        p = np.reshape(position, [-1, 2])
        dist = np.sqrt(np.sum(np.square(p - all_data), axis=1))
        return np.mean(np.sort(dist)[:15])


class Policy:
    def __init__(self, hp, params=None):
        # 根据parameter初始化
        self.hp = hp
        self.input_dim, self.output_dim, self.hidden_size = hp.input_dim, hp.output_dim, hp.hidden_size
        self.param_count = hp.input_dim * hp.hidden_size + self.hidden_size + self.hidden_size * self.hidden_size + \
                           self.hidden_size + self.hidden_size * self.output_dim + self.output_dim
        if params is not None:
            assert len(params) == self.param_count
        self.params = params
        self.ddpg_agent = self.hp.ddpg_agent

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
        output_action = tf.nn.tanh(tf.matmul(l2, param_list[4]) + param_list[5])
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
            self.hp.replay_buffer.add(
                (obs, next_obs, action, reward, done))  # saving the experience replay for each path
            if np.sqrt(np.sum(np.square(next_obs[:2] - TARGET_GOAL))) < 0.1:
                reward = 0
            else:
                reward = -1
            # reward = -np.sqrt(np.sum(np.square(next_obs[:2] - target_goal)))
            obs = next_obs
            total_reward += reward
            if done:
                break
        if len(self.hp.archive) > 0:
            novelty = self.hp.cal_novelty(obs[:2])
        else:
            novelty = 0
        return novelty, total_reward, obs[:2]

    def ddpg_update(self, goal):
        # giving a specific goal, update the policy to reach the goal as quick as possible using the experience replay
        self.ddpg_agent.syn_params(self.get_detail_params())
        self.ddpg_agent.set_replay_buffer(deepcopy(self.hp.replay_buffer))
        a_loss, c_loss = self.ddpg_agent.train_to_reach(self.hp.learning_step, goal, self.hp.explore_range)
        new_params_list = self.ddpg_agent.get_params()
        new_params = np.hstack([_.flatten() for _ in new_params_list])
        self.params = new_params
        return a_loss, c_loss


env = AntMazeEnv(maze_id=TASK_NAME)
RESTART = False
restart_file_name = 'novelty_search_final_population'
episode_num = EPISODE_NUMBER
seed = 1
num_worst = REPLACE

hp = HP(env=env, input_dim=30, output_dim=8, seed=seed)
policy = Policy(hp)
ga = GA(num_params=policy.get_params_count(), pop_size=POPULATION, elite_frac=0.1, mut_rate=0.9)

all_data = []
final_pos = []

for episode in range(episode_num):
    start_time = datetime.datetime.now()
    if RESTART:
        population = pickle.load(open(restart_file_name, mode='rb'))
    else:
        population = ga.ask()
    reward = []
    novelty = []
    position = []
    for p in population:
        policy.set_params(p)
        n, r, last_position = policy.get_fitness()
        novelty.append(n)
        reward.append(r)
        position.append(last_position)
    fitness = novelty
    for p in position:
        policy.hp.archive.append(p)  # update novelty archive
    ga.tell(population, fitness)
    best_index = np.argmax(fitness)
    # 列出最差的几个agent
    worst_index = np.argsort(fitness)[:num_worst]
    # 对最差的几个进行重新学习
    actor_loss = []
    critic_loss = []
    distance_to_goal = []
    new_bc = []
    origin_novelty = []
    improved_novelty = []
    current_goal = position[best_index]
    initial_bc = position
    for idx in worst_index:
        policy.set_params(population[idx])
        a_loss, c_loss = policy.ddpg_update(goal=position[best_index])
        t_fitness, _, bc = policy.get_fitness()
        # if t_fitness > fitness[idx]:
        ga.add_v2(index=idx, parameters=policy.params, fitness=t_fitness)
        new_bc.append(bc)
        policy.hp.archive.append(bc)
        print('Original novelty:', fitness[idx])
        origin_novelty.append(fitness[idx])
        print('improved novelty:', t_fitness)
        improved_novelty.append(t_fitness)
        print('Distance to current goal:', np.sqrt(np.sum(np.square(position[best_index] - bc))))
        actor_loss.append(a_loss)
        critic_loss.append(c_loss)
        distance_to_goal.append(np.sqrt(np.sum(np.square(position[best_index] - bc))))
        print('Actor loss:', a_loss)
        print('Critic loss:', c_loss)

    all_data.append(
        {'episode': episode, 'best_fitness': fitness[best_index], 'best_reward': reward[best_index],
         'original_novelty': origin_novelty, 'improved_novelty': improved_novelty, 'distance_to_goal': distance_to_goal,
         'actor_loss': actor_loss, 'critic_loss': critic_loss, 'current_goal': current_goal, 'new_bc': new_bc,
         'initial_bc': initial_bc, 'novelty': fitness},
    )
    final_pos.append(position[best_index])
    print('######' + TASK_NAME)
    print('Episode ', episode)
    print('Best fitness value: ', fitness[best_index])
    print('Best reward: ', reward[best_index])
    print('Final distance: ', np.sqrt(np.sum(np.square(position[best_index] - TARGET_GOAL))))
    print('Running time:', (datetime.datetime.now() - start_time).seconds)
pickle.dump(all_data, open(TASK_NAME + '_garl_reward_v' + VERSION, mode='wb'))
pickle.dump(final_pos, open(TASK_NAME + '_garl_final_pos_v' + VERSION, mode='wb'))
# pickle.dump(population, open('garl_final_population_v1_10', mode='wb'))
