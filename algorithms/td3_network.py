import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import pprint as pp
import gym
from algorithms.utils import *
import os


class Actor(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, hidden_size,
                 namescope='default'):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.namescope = namescope

        self.inp = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32, name=self.namescope + '_inp')  # 输入state

        self.out, self.scaled_out = self.create_actor_network(self.namescope + 'main_actor')  # 输出动作，和输入的状态

        self.network_params = tf.trainable_variables()

        self.target_out, self.target_scaled_out = self.create_actor_network(
            self.namescope + 'target_actor')  # 创建targetnetwork

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]  # 按照创建的顺序构建的

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.w1 = tf.placeholder(shape=[self.s_dim, self.a_dim], dtype=tf.float32)
        self.b1 = tf.placeholder(shape=[self.a_dim, ], dtype=tf.float32)
        self.input_parameters = [self.w1, self.b1]
        self.syn_target_op = [self.target_network_params[i].assign(self.input_parameters[i]) for i in
                              range(len(self.target_network_params))]
        self.syn_policy_op = [self.network_params[i].assign(self.input_parameters[i]) for i in
                              range(len(self.target_network_params))]

    def create_actor_network(self, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            net = self.inp
            net = slim.fully_connected(net, self.a_dim, activation_fn=None,
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
            scaled_out = tf.multiply(net, self.action_bound)
            return net, scaled_out

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inp: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.inp: inputs
        })


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, inp_actions, hidden_size, namescope='default'):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.inp_actions = inp_actions
        self.hidden_size = hidden_size
        self.namescope = namescope

        self.inp = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float32)

        self.total_out, _ = self.create_critic_network(self.namescope + 'main_critic', self.inp_actions)
        self.out1, self.out2 = self.create_critic_network(self.namescope + 'main_critic', self.action,
                                                          reuse=True)  # 要重用里面的变量，所以要设为true,创建时参数一样

        self.target_out1, self.target_out2 = self.create_critic_network(self.namescope + 'target_critic', self.action)

        self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.namescope + 'main_critic')
        self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                       self.namescope + 'target_critic')  # 得到target的variables
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.reduce_mean(tf.square(self.out1 - self.predicted_q_value)) + tf.reduce_mean(
            tf.square(self.out2 - self.predicted_q_value))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.network_params)

    def create_critic_network(self, scope, actions, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            net = tf.concat([self.inp, actions], axis=1)
            net = slim.fully_connected(net, 1, activation_fn=None)
            net1 = net
            net = tf.concat([self.inp, actions], axis=1)
            net = slim.fully_connected(net, 1, activation_fn=None)
            net2 = net
        return net1, net2

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def predict1(self, inputs, action):
        return self.sess.run(self.out1, feed_dict={
            self.inp: inputs,
            self.action: action
        })

    def predict2(self, inputs, action):
        return self.sess.run(self.out2, feed_dict={
            self.inp: inputs,
            self.action: action
        })

    def predict_target1(self, inputs, action):
        return self.sess.run(self.target_out1, feed_dict={
            self.inp: inputs,
            self.action: action
        })

    def predict_target2(self, inputs, action):
        return self.sess.run(self.target_out2, feed_dict={
            self.inp: inputs,
            self.action: action
        })


class TD3(object):
    def __init__(self, state_dim, action_dim, action_bound, discount_factor=1,
                 seed=1, actor_lr=1e-3, critic_lr=1e-3, batch_size=100, namescope='default',
                 tau=0.005, policy_noise=0.1, noise_clip=0.5, hidden_size=300):
        np.random.seed(int(seed))
        tf.set_random_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        # env.seed(int(seed))
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.hidden_size = hidden_size
        self.actor = Actor(self.sess, state_dim, action_dim, action_bound,
                           actor_lr, tau, int(batch_size), self.hidden_size, namescope=namescope + str(seed))
        self.critic = Critic(self.sess, state_dim, action_dim, critic_lr, tau,
                             self.actor.scaled_out, self.hidden_size, namescope=namescope + str(seed))
        actor_loss = -tf.reduce_mean(self.critic.total_out)
        self.actor_train_step = tf.train.AdamOptimizer(actor_lr).minimize(actor_loss,
                                                                          var_list=self.actor.network_params)
        self.action_bound = action_bound
        self.sess.run(tf.global_variables_initializer())
        self.replay_buffer = ReplayBuffer()

    def train(self, iterations):
        if self.replay_buffer.get_size() < 1000:
            return
        for i in range(iterations):
            state, next_state, action, reward, done = self.replay_buffer.sample(self.batch_size)
            noise = np.random.normal(0, self.policy_noise, size=[self.batch_size, self.action_dim])
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            temp_action = np.reshape(self.actor.predict_target(next_state), [self.batch_size, -1])
            next_action = temp_action + noise
            next_action = np.clip(next_action, -self.action_bound, self.action_bound)
            target_q1 = self.critic.predict_target1(next_state, next_action)
            target_q2 = self.critic.predict_target2(next_state, next_action)
            target_q = np.minimum(target_q1, target_q2)

            y_i = reward + (1 - done) * self.discount_factor * target_q

            self.sess.run(self.critic.train_step, feed_dict={self.critic.inp: state,
                                                             self.critic.action: np.reshape(action, [self.batch_size,
                                                                                                     self.action_dim]),
                                                             self.critic.predicted_q_value: np.reshape(y_i, [-1, 1])})

            if i % 2 == 0:
                self.sess.run(self.actor_train_step, feed_dict={self.actor.inp: state, self.critic.inp: state})
                self.actor.update_target_network()
                self.critic.update_target_network()

    def get_action(self, s):
        return self.actor.predict(np.reshape(s, (1, self.actor.s_dim)))

    def store(self, s, s2, action, r, done_bool):
        self.replay_buffer.add((s, s2, action, r, done_bool))

    def get_params(self):
        return self.sess.run(self.actor.network_params[:len(self.actor.network_params)])

    def get_action_target(self, s):
        return self.actor.predict_target(np.reshape(s, (1, self.actor.s_dim)))

    def syn_params(self, params_):
        self.sess.run(self.actor.syn_target_op, feed_dict={self.actor.w1: params_[0],
                                                           self.actor.b1: params_[1]})
        self.sess.run(self.actor.syn_policy_op, feed_dict={self.actor.w1: params_[0],
                                                           self.actor.b1: params_[1]})
