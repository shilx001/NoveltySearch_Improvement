import numpy as np
import gym
import collections
import datetime
import tensorflow as tf
# from algorithms.td3_network import *
from ant_maze_env import *
import pickle


# from td3_network import *


# neural networks as policy, with adam optimizer
# dynamically adjust the weight
# Policy discrepency as bc
# add td3 update

class HP:
    # hyper parameters
    def __init__(self, env, input_size=None, output_size=None, total_episodes=1000,
                 episode_length=1000, learning_rate=0.01, weight=1, action_bound=1,
                 num_samples=4, noise=0.02, bc_index=[0], std_dev=0.03, batch_size=64,
                 meta_population_size=3, seed=1, normalizer=None, hidden_size=64,
                 learning_steps=100, network_lr=1e-3, tau=0.005):
        self.env = env
        np.random.seed(seed)
        self.env.seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.total_episodes = total_episodes
        self.episode_length = episode_length
        self.lr = learning_rate
        self.num_samples = num_samples
        self.noise = noise
        self.meta_population_size = meta_population_size
        self.seed = seed
        self.bc_index = bc_index
        self.weight = weight
        self.normalizer = normalizer
        self.hidden_size = hidden_size
        self.stddev = std_dev
        self.learning_steps = learning_steps
        self.network_lr = network_lr
        self.tau = tau
        self.update_coefficient = 0.5
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.sess = tf.Session()
        # self.replay_buffer = utils.ReplayBuffer()
        # self.actor = Actor(self.sess, self.input_size, self.output_size, self.action_bound, self.network_lr,
        #                   self.tau, self.batch_size)
        # self.critic = Critic(self.sess, self.input_size, self.output_size, self.network_lr, self.tau,
        #                     self.actor.scaled_out)
        # self.actor_loss = -tf.reduce_mean(self.critic.total_out)
        # self.actor_train_step = tf.train.AdamOptimizer(self.network_lr).minimize(self.actor_loss)
        # self.sess.run(tf.global_variables_initializer())


class Archive:
    # the archive, store the behavior
    def __init__(self, number_neighbour=15):
        self.data = []
        self.k = number_neighbour

    def add_policy(self, policy_bc):
        # 只存policy的bc
        self.data.append(policy_bc)

    def initialize(self, meta_population):
        for policy in meta_population.population:
            policy.evaluate()
            self.data.append(policy.bc)

    def novelty(self, policy_bc):
        # calculate the novelty of policy
        dist = np.sort(np.sum(np.square(policy_bc - np.array(self.data)), axis=1), axis=None)
        return np.mean(dist[:self.k])


class Normalizer:
    # Normalizes the input observations
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):  # observe a space, dynamic implementation of calculate mean of variance
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)  # 方差，清除了小于1e-2的

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


class Adam_optimizer:
    def __init__(self, lr):
        self.m_t = 0
        self.v_t = 0
        self.t = 0
        self.alpha = lr
        self.beta_1 = 0.9
        self.beta_2 = 0.99
        self.epsilon = 1e-8

    def update(self, g_t):
        self.t += 1
        self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * g_t
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * (g_t * g_t)
        m_cap = self.m_t / (1 - (self.beta_1 ** self.t))
        v_cap = self.v_t / (1 - (self.beta_2 ** self.t))
        return (self.alpha * m_cap) / (np.sqrt(v_cap) + self.epsilon)


class Policy:
    def __init__(self, hp):
        # behavior characteristic初始化为None
        self.hp = hp
        # 针对每个层
        self.w1 = np.random.randn(self.hp.input_size, self.hp.hidden_size) * self.hp.stddev
        self.b1 = np.zeros([self.hp.hidden_size, ])
        self.w2 = np.random.randn(self.hp.hidden_size, self.hp.hidden_size) * self.hp.stddev
        self.b2 = np.zeros([self.hp.hidden_size, ])
        self.w3 = np.random.randn(self.hp.hidden_size, self.hp.output_size) * self.hp.stddev
        self.b3 = np.zeros([self.hp.output_size, ])
        self.bc = None
        # used for adam update
        self.wa_1 = Adam_optimizer(self.hp.lr)
        self.wa_2 = Adam_optimizer(self.hp.lr)
        self.wa_3 = Adam_optimizer(self.hp.lr)
        self.ba_1 = Adam_optimizer(self.hp.lr)
        self.ba_2 = Adam_optimizer(self.hp.lr)
        self.ba_3 = Adam_optimizer(self.hp.lr)

    def get_action(self, state, delta=None):
        state = np.reshape(state, [1, self.hp.input_size])
        if delta is None:
            output1 = np.maximum(np.dot(state, self.w1) + self.b1, 0)
            output2 = np.maximum(np.dot(output1, self.w2) + self.b2, 0)
            action = np.tanh(np.reshape(np.dot(output2, self.w3) + self.b3, [self.hp.output_size, ]))
        else:
            output1 = np.maximum(np.dot(state, self.w1 + delta[0]) + self.b1 + delta[1], 0)
            output2 = np.maximum(np.dot(output1, self.w2 + delta[2]) + self.b2 + delta[3], 0)
            action = np.tanh(
                np.reshape(np.dot(output2, self.w3 + delta[4]) + self.b3 + delta[5], [self.hp.output_size, ]))
        return action * self.hp.action_bound

    def evaluate(self, delta=None):
        # 根据当前state执在环境中执行一次，返回获得的reward和novelty
        # env为环境，为了防止多次初始化这里传入环境
        total_reward = 0
        obs = self.hp.env.reset()
        for i in range(self.hp.episode_length):
            action = np.clip(self.get_action(obs, delta=delta), -30, 30)
            next_obs, reward, done, _ = self.hp.env.step(action)
            obs = next_obs
            total_reward += reward
            if done:
                break
        # 计算bc并更新
        if delta is None:
            self.bc = obs[self.hp.bc_index]
        return total_reward, obs[self.hp.bc_index]

    def update(self, rollouts, sigma_rewards):
        step = 0
        # 针对每个参数进行更新
        for r, delta in rollouts:
            step += r * delta[0]
        self.w1 += self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        step = 0
        for r, delta in rollouts:
            step += r * delta[1]
        self.b1 += self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        step = 0
        for r, delta in rollouts:
            step += r * delta[2]
        self.w2 += self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        step = 0
        for r, delta in rollouts:
            step += r * delta[3]
        self.b2 += self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        step = 0
        for r, delta in rollouts:
            step += r * delta[4]
        self.w3 += self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        step = 0
        for r, delta in rollouts:
            step += r * delta[5]
        self.b3 += self.hp.lr / (self.hp.num_samples * sigma_rewards) * step

    def adam_update(self, rollouts, sigma_rewards):
        step = 0
        for r, delta in rollouts:
            step += r * delta[0]
        grad = self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        self.w1 += self.wa_1.update(grad)
        step = 0
        for r, delta in rollouts:
            step += r * delta[1]
        grad = self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        self.b1 += self.ba_1.update(grad)
        step = 0
        for r, delta in rollouts:
            step += r * delta[2]
        grad = self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        self.w2 += self.wa_2.update(grad)
        step = 0
        for r, delta in rollouts:
            step += r * delta[3]
        grad = self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        self.b2 += self.ba_2.update(grad)
        step = 0
        for r, delta in rollouts:
            step += r * delta[4]
        grad = self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        self.w3 += self.wa_3.update(grad)
        step = 0
        for r, delta in rollouts:
            step += r * delta[5]
        grad = self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        self.b3 += self.ba_3.update(grad)

    def sample_deltas(self):
        return [np.random.randn(*self.w1.shape) * self.hp.noise,
                np.random.randn(*self.b1.shape) * self.hp.noise,
                np.random.randn(*self.w2.shape) * self.hp.noise,
                np.random.randn(*self.b2.shape) * self.hp.noise,
                np.random.randn(*self.w3.shape) * self.hp.noise,
                np.random.randn(*self.b3.shape) * self.hp.noise]


class MetaPopulation:
    def __init__(self, population_size):
        self.population_size = population_size
        self.population = collections.deque(maxlen=population_size)

    def sample(self, archive):
        # sample a policy with specific probability
        # 计算当前所有的概率，并选择一个policy, 返回当前policy的index和值
        novelty = [archive.novelty(policy.bc) for policy in self.population]
        p = np.array(novelty) / np.sum(novelty)
        index = int(np.random.choice(self.population_size, 1, p=np.array(novelty) / np.sum(novelty)))
        return self.population[index], index

    def initialize(self, hp):
        for i in range(self.population_size):
            policy = Policy(hp)
            policy.evaluate()
            self.population.append(policy)

    def update(self, index, policy):
        # 新的policy 替换掉旧的
        self.population[index] = policy


class NoveltySearch:
    def __init__(self, hp):
        self.hp = hp
        self.archive = Archive()

    def train(self):
        weight = 1
        meta_population = MetaPopulation(population_size=self.hp.meta_population_size)
        meta_population.initialize(self.hp)
        self.archive.initialize(meta_population)
        reward_memory = []
        novelty_memory = []
        for t in range(self.hp.total_episodes):
            start_time = datetime.datetime.now()
            policy, index = meta_population.sample(self.archive)
            deltas = [policy.sample_deltas() for _ in
                      range(self.hp.num_samples)]
            novelty_forward_list = []
            novelty_backward_list = []
            forward_reward_list = []
            backward_reward_list = []
            bc_forward_list=[]
            bc_backward_list=[]
            for i in range(self.hp.num_samples):
                delta = deltas[i]
                reward_forward, bc_forward = policy.evaluate(delta)
                neg_delta = [-delta[_] for _ in range(len(delta))]
                reward_backward, bc_backward = policy.evaluate(neg_delta)
                forward_reward_list.append(reward_forward)
                # self.archive.add_policy(bc_backward)
                # self.archive.add_policy(bc_forward)
                backward_reward_list.append(reward_backward)
                novelty_forward = self.archive.novelty(bc_forward)
                novelty_backward = self.archive.novelty(bc_backward)
                novelty_forward_list.append(novelty_forward)
                novelty_backward_list.append(novelty_backward)
                bc_forward_list.append(bc_forward)
                bc_backward_list.append(bc_backward)
            rollouts = [((forward_reward_list[j] - backward_reward_list[j]) *
                         (1 - self.hp.weight) + (novelty_forward_list[j] - novelty_backward_list[j]) * self.hp.weight,
                         deltas[j]) for j in range(self.hp.num_samples)]
            for bc in bc_forward_list:
                self.archive.add_policy(bc)
            for bc in bc_backward_list:
                self.archive.add_policy(bc)
            # sigma_rewards = np.std(
            #    (1 - weight) * np.array(forward_reward_list + backward_reward_list) + np.array(
            #        novelty_forward_list + novelty_backward_list) * (weight))
            # policy.update(rollouts, sigma_rewards)
            policy.adam_update(rollouts, 1)
            meta_population.update(index, policy)
            test_reward, bc = policy.evaluate()
            '''
            if test_reward > max_fitness:
                max_fitness = test_reward
                weight = np.minimum(1, weight + 0.05)
                t_best = 0
            else:
                t_best += 1
            if t_best >= t_w:
                weight = max(0, weight - 0.05)
                t_best = 0
            '''
            test_novelty = self.archive.novelty(bc)
            print('#######')
            print('Episode ', t)
            # print('Total reward is: ', test_reward)
            print('Novelty is : ', test_novelty)
            print('Running time:', (datetime.datetime.now() - start_time).seconds)
            reward_memory.append(test_reward)
            novelty_memory.append(test_novelty)
        return reward_memory, novelty_memory


env = AntMazeEnv(maze_id='Maze')
hp = HP(env=env, input_size=30, output_size=8, total_episodes=1000, episode_length=500, action_bound=30,
        bc_index=[0, 1], seed=1, weight=0, learning_rate=0.02, num_samples=100, noise=0.03, meta_population_size=3)
trainer = NoveltySearch(hp)
reward, novelty = trainer.train()
pickle.dump(trainer.archive.data, open('novelty_search_es_data', 'wb'))
pickle.dump(novelty, open('novelty_search_es_novelty', 'wb'))
