import numpy as np
from GA import *
import gym
import tensorflow as tf
import datetime
from ant_maze_env import *
from algorithms.td3_network import *
import pickle


# add pre-training of the lower policy
class HP:
    def __init__(self, env, seed=1, input_dim=30, output_dim=2, action_bound=20, hidden_size=64, pop_size=10):
        # HP只是meta-policy的！
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
        self.param_count = hp.input_dim * hp.hidden_size + self.hidden_size + self.hidden_size * self.hidden_size + \
                           self.hidden_size + self.hidden_size * self.output_dim + self.output_dim
        if params is not None:
            assert len(params) == self.param_count
        self.params = params

    def get_params_count(self):
        return self.param_count

    def set_params(self, params):
        self.params = params

    def evaluate_tf(self, state):
        # 得出action值，三层神经网络？用parameter来说
        sess = tf.Session()
        input_state = np.reshape(state, [1, self.input_dim])
        feed_state = tf.placeholder(dtype=tf.float64, shape=[1, self.input_dim])
        param_list = self.get_detail_params()
        l1 = tf.nn.relu(tf.matmul(feed_state, param_list[0]) + param_list[1])
        l2 = tf.nn.relu(tf.matmul(l1, param_list[2])) + param_list[3]
        output_action = tf.nn.tanh(tf.matmul(l2, param_list[4]) + param_list[5])
        return sess.run(output_action, feed_dict={feed_state: input_state}) * self.hp.action_bound

    def evaluate(self, state):
        input_state = np.reshape(state, [1, self.input_dim])
        param_list = self.get_detail_params()
        l1 = np.maximum(0, input_state.dot(param_list[0]) + param_list[1])
        l2 = np.maximum(0, l1.dot(param_list[2]) + param_list[3])
        return np.tanh(l2.dot(param_list[4]) + param_list[5]) * self.hp.action_bound

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
        self.optimizer = GA(num_params=self.h_policy.get_params_count(), pop_size=hp.pop_size, elite_frac=0.1)

    def pre_train(self, num_episodes):
        # 训练n个episodes来获得神经网络
        step = 0
        done = False
        env = self.hp.env
        print('Begin pre-train:')
        for t in range(num_episodes):
            obs = env.reset()
            while not done and step < 500:
                if step % 10 is 0:
                    current_goal = np.clip(np.random.randn(2) * 5, -10, 10)
                experience_replay_buffer = []
                for i in range(10):
                    state_goal = np.hstack((obs.flatten(), current_goal))
                    action = np.clip(self.l_policy.get_action(state_goal) + np.random.randn(8, ) * 10, -30,
                                     30)  # evaluate primitive action
                    next_obs, reward, done, _ = env.step(action)
                    #l_reward = -1 * (np.sqrt(np.sum(np.square(obs[:2] + current_goal - next_obs[:2]))) - 0.01 > 0)
                    l_reward = -np.sqrt(np.sum(np.square(obs[:2] + current_goal - next_obs[:2])))
                    l_done = l_reward == 0
                    next_state_goal = np.hstack((next_obs.flatten(), current_goal))
                    self.l_policy.store(state_goal, next_state_goal, action, l_reward, l_done)
                    experience_replay_buffer.append((obs, action, next_obs))
                    obs = next_obs
                    step += 1
                    if l_done is 0:
                        print('############Reach subgoal!')
                        break
                final_replay = experience_replay_buffer[-1]  # 最后一个transition
                final_goal = (final_replay[2] - final_replay[0])[:2]
                for s1, a, s2 in experience_replay_buffer:  # 针对每个来说
                    state_goal = np.hstack((s1.flatten(), final_goal))
                    next_state_goal = np.hstack((s2.flatten(), final_goal))
                    r = -1 * (np.sqrt(np.sum(np.square(s1[:2] + final_goal - s2[:2]))) - 0.01 > 0)
                    d = r == 0
                    self.l_policy.store(state_goal, next_state_goal, a, r, d)
            a_loss, c_loss = self.l_policy.train(10)
            print('Step ', t)
            print('Actor loss:', a_loss)
            print('Critic loss:', c_loss)
        print('End pre-train')

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
            l_reward = -1 * (np.sqrt(np.sum(np.square(obs[:2] + current_goal - next_obs[:2]))) - 0.1 > 0)
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
        a_loss, c_loss = self.l_policy.train(5)
        final_pos = obs[:2]
        return h_reward, final_pos, a_loss, c_loss

    def get_fitness_v2(self, params):
        # 根据h_policy的params得到fitness, for 1 rollout.
        self.h_policy.set_params(params)
        env = self.hp.env
        obs = env.reset()
        h_reward = 0
        target_goal = np.array([0, 16])
        step = 0
        done = False
        penalty = 0
        while step < 500 and not done:
            if step % 10 is 0:  # 每十次重设一下goal
                current_goal = self.h_policy.evaluate(obs).flatten()
            # print('Goal is :', current_goal)
            experience_replay_buffer = []
            for i in range(10):  # 每10个测试一下
                state_goal = np.hstack((obs.flatten(), current_goal))
                action = self.l_policy.get_action(state_goal)  # evaluate primitive action
                next_obs, reward, done, _ = env.step(action)
                step += 1
                l_reward = -1 * (np.sqrt(np.sum(np.square(obs[:2] + current_goal - next_obs[:2]))))
                l_done = l_reward > -0.1
                next_state_goal = np.hstack((next_obs.flatten(), current_goal))
                self.l_policy.store(state_goal, next_state_goal, action, l_reward, l_done)  # normal experience replay
                experience_replay_buffer.append((obs, action, next_obs))
                obs = next_obs
                if np.sqrt(np.sum(np.square(next_obs[:2] - target_goal))) < 0.1:
                    h_reward += 0
                else:
                    h_reward += -1
                if l_done is 0:
                    break
            if not l_done:
                penalty -= 100
            if np.sqrt(np.sum(np.square(next_obs[:2] - target_goal))) < 0.1:
                break
        # a_loss, c_loss = self.l_policy.train(10)
        a_loss = 0
        c_loss = 0
        final_pos = obs[:2]
        return h_reward, final_pos, a_loss, c_loss, penalty

    def train(self, num_generation):
        fitness_list = []
        final_pos_list = []
        actor_loss = []
        critic_loss = []
        self.pre_train(1000)
        for episode in range(num_generation):
            start_time = datetime.datetime.now()
            population = self.optimizer.ask()
            fitness = []
            final_pos = []
            novelty = []
            for i in range(self.hp.pop_size):
                f, p, a, c, penalty = self.get_fitness_v2(params=population[i])
                fitness.append(f)
                final_pos.append(p)
                novelty.append(self.hp.cal_novelty(p))
                # print('Penalty is:',penalty)
                actor_loss.append(a)
                critic_loss.append(c)
            for i in final_pos:  # 加入archive
                self.hp.archive.append(i)
            self.optimizer.tell(population, novelty)
            best_idx = np.argmax(novelty)
            fitness_list.append(max(novelty))
            final_pos_list.append(final_pos[best_idx])
            print('######')
            print('Episode ', episode)
            print('Best fitness value: ', max(fitness))
            print('Best novelty: ', novelty[best_idx])
            print('Total penalty: ', penalty)
            print('Final distance:', np.sqrt(np.sum(np.square(final_pos[best_idx] - np.array([0, 16])))))
            print('Running time:', (datetime.datetime.now() - start_time).seconds)
            print('Actor mean loss:', np.mean(actor_loss[-10:]))
            print('Critic mean loss:', np.mean(critic_loss[-10:]))
        return fitness_list, final_pos_list, actor_loss, critic_loss


env = AntMazeEnv(maze_id='Maze')
hp = HP(env=env)
meta_policy = MetaPolicy(hp)
fitness, final_pos, actor_loss, critic_loss = meta_policy.train(1000)
pickle.dump(fitness, open('hga_novelty_fitness', mode='wb'))
pickle.dump(final_pos, open('hga_novelty_final_pos', mode='wb'))
pickle.dump(actor_loss, open('hga_novelty_actor_loss', mode='wb'))
pickle.dump(critic_loss, open('hga_novelty_critic_loss', mode='wb'))
