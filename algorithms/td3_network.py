import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import pprint as pp
import gym
import utils
import os


class Actor(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, hidden_size=64):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.inp = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)  # 输入state

        self.out, self.scaled_out = self.create_actor_network('main_actor')  # 输出动作，和输入的状态

        self.network_params = tf.trainable_variables()

        self.target_out, self.target_scaled_out = self.create_actor_network('target_actor')  # 创建targetnetwork

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]  # 按照创建的顺序构建的

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.w1 = tf.placeholder(shape=[self.s_dim, self.hidden_size], dtype=tf.float32)
        self.b1 = tf.placeholder(shape=[self.hidden_size, ], dtype=tf.float32)
        self.w2 = tf.placeholder(shape=[self.hidden_size, self.hidden_size], dtype=tf.float32)
        self.b2 = tf.placeholder(shape=[self.hidden_size, ], dtype=tf.float32)
        self.w3 = tf.placeholder(shape=[self.hidden_size, self.a_dim], dtype=tf.float32)
        self.b3 = tf.placeholder(shape=[self.a_dim, ], dtype=tf.float32)
        self.input_parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        self.refresh_target_op = [self.target_network_params[i].assign(self.input_parameters[i]) for i in
                                  range(len(self.target_network_params))]
        self.refresh_policy_op = [self.network_params[i].assign(self.input_parameters[i]) for i in
                                  range(len(self.target_network_params))]

    def create_actor_network(self, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            net = self.inp
            net = slim.fully_connected(net, self.hidden_size, activation_fn=tf.nn.relu,
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
            net = slim.fully_connected(net, self.hidden_size, activation_fn=tf.nn.relu)
            net = slim.fully_connected(net, self.a_dim, activation_fn=tf.nn.tanh)
            scaled_out = tf.multiply(net, self.action_bound)
            return net, scaled_out

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inp: inputs
        })

    def refresh_params(self, params_):
        self.sess.run(self.refresh_target_op, feed_dict={self.w1: params_[0],
                                                         self.b1: params_[1],
                                                         self.w2: params_[2],
                                                         self.b2: params_[3],
                                                         self.w3: params_[4],
                                                         self.b3: params_[5]})
        self.sess.run(self.refresh_policy_op, feed_dict={self.w1: params_[0],
                                                         self.b1: params_[1],
                                                         self.w2: params_[2],
                                                         self.b2: params_[3],
                                                         self.w3: params_[4],
                                                         self.b3: params_[5]})

    def get_params(self):
        return self.sess.run(self.target_network_params)

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.inp: inputs
        })


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, inp_actions, hidden_size=64):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.inp_actions = inp_actions
        self.hidden_size = hidden_size

        self.inp = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float32)

        self.total_out, _ = self.create_critic_network('main_critic', self.inp_actions)
        self.out1, self.out2 = self.create_critic_network('main_critic', self.action,
                                                          reuse=True)  # 要重用里面的变量，所以要设为true,创建时参数一样

        self.target_out1, self.target_out2 = self.create_critic_network('target_critic', self.action)

        self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main_critic')
        self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                       'target_critic')  # 得到target的variables
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.reduce_mean(tf.square(self.out1 - self.predicted_q_value)) + tf.reduce_mean(
            tf.square(self.out2 - self.predicted_q_value))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                              var_list=self.network_params)

    def create_critic_network(self, scope, actions, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            net = tf.concat([self.inp, actions], axis=1)
            net = slim.fully_connected(net, self.hidden_size)
            net = slim.fully_connected(net, self.hidden_size)
            net = slim.fully_connected(net, 1, activation_fn=None)
            net1 = net
            net = tf.concat([self.inp, actions], axis=1)
            net = slim.fully_connected(net, self.hidden_size)
            net = slim.fully_connected(net, self.hidden_size)
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
