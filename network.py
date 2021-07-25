# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:20:16 2020

@author: golbang
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.contrib.layers as layer
import numpy as np

def elu(x):
    return tf.nn.elu(x)+1

class policy:
    def __init__(self, sess, num_of_feature=4, day_length=50, num_of_asset=10,  filter_size = 3, learning_rate = 1e-4, regularizer_rate = 1e-3, gamma = 0.03, name='allocator'):
        self.sess = sess
        self.input_size=day_length
        self.output_size=num_of_asset
        self.net_name=name
        self.num_of_feature = num_of_feature
        self.filter_size = filter_size
        self.gamma = gamma
        regularizer = tf.contrib.layers.l2_regularizer(scale=regularizer_rate)
        initializer = layer.xavier_initializer()
        
        self._X=tf.placeholder(tf.float64,[None,self.output_size,self.input_size,self.num_of_feature], name = "s") # shape: batch,10,50,4
        self._r=tf.placeholder(tf.float64,[None,self.output_size], name = 'r')
        self._H = tf.placeholder(tf.float64, [None,self.output_size,self.output_size],name = 'cov')
        
        self.conv1 = layer.conv2d(self._X, 8, [1,self.filter_size], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer)
        self.conv2 = layer.conv2d(self.conv1, 1, [1,self.input_size-self.filter_size+1], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer, weights_regularizer = regularizer)
        
        self.fc1 = layer.fully_connected(layer.flatten(self.conv2), 50, activation_fn=tf.nn.leaky_relu, weights_regularizer = regularizer)

        self.policy = layer.fully_connected(self.fc1, self.output_size, activation_fn=tf.nn.softmax)
        
        self.WH = tf.matmul(tf.reshape(self.policy,(-1,1,self.output_size)),self._H)
        self.WHW = tf.matmul(self.WH,tf.reshape(self.policy,(-1,self.output_size,1)))
        self.sigma_p = tf.sqrt(self.WHW)
        
        self.risk_loss = ((self.sigma_p-self.gamma)*1e+2)**2
        self.risk_loss = tf.reshape(self.risk_loss,(-1,1))
        
        self.weighted_reward = tf.reduce_sum((self._r*10) * self.policy,1)
        
        self.loss = tf.reduce_sum(-self.weighted_reward + self.risk_loss)
        
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self, s):
        s = np.expand_dims(s,axis=0)
        self.weight=self.sess.run(self.policy, {self._X: s})
        return self.weight

    def update(self, episode_memory):
        episode_memory = np.array(episode_memory)
        s = np.array(episode_memory[:,0].tolist())
        r = np.array(episode_memory[:,1].tolist())
        cov = np.array(episode_memory[:,2].tolist())[:,0]
        a_loss = self.sess.run([self.loss,self.train], {self._X: s, self._r: r, self._H : cov})[0]
        return a_loss
    
class estimator:
    def __init__(self, sess, num_of_feature=4, day_length=50, num_of_asset=10,  filter_size = 3, learning_rate = 1e-3, name='estimator'):
        self.sess = sess
        self.input_size=day_length
        self.net_name=name
        self.num_of_feature = num_of_feature
        self.filter_size = filter_size
        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-3)
        initializer = layer.xavier_initializer()
        
        self._X=tf.placeholder(tf.float64,[None,num_of_asset, self.input_size,self.num_of_feature], name = "x") # shape: batch,10,50,4
        self._z = tf.placeholder(tf.float64,[None, 1, num_of_asset],name = 'z')
        
        self.conv1 = layer.conv2d(self._X, 8, [1,self.filter_size], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer)
        self.conv2 = layer.conv2d(self.conv1, 1, [1,self.input_size-self.filter_size+1], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer)
        
        self.fc1 = layer.fully_connected(layer.flatten(self.conv2), 50, activation_fn=tf.nn.sigmoid)
        
        self.fc_L = layer.fully_connected(self.fc1, num_of_asset * num_of_asset, activation_fn=tf.nn.tanh)
        self.L = tf.reshape(self.fc_L,(-1,num_of_asset,num_of_asset))
        
        self.R = tf.matmul(self.L,tf.transpose(self.L,perm = [0,2,1]))+1e-3
        
        self.R_inv = tf.linalg.inv(self.R)
        
        self.multiply_inv = tf.reduce_sum(tf.matmul(tf.matmul(self._z,self.R_inv),tf.transpose(self._z,perm = [0,2,1])))
        self.logdet = tf.log(tf.linalg.det(self.R))
        
        self.loss = tf.reduce_sum(self.logdet+self.multiply_inv)
        
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self, x):
        x = np.expand_dims(x,axis=0)
        covariation = self.sess.run(self.R, {self._X: x})
        return covariation

    def update(self, episode_memory):
        self.episode_memory = np.array(episode_memory)
        x = np.array(self.episode_memory[:,0].tolist())
        z = np.array(self.episode_memory[:,1].tolist())
        c_loss = self.sess.run([self.loss,self.train], {self._X: x, self._z: z})[0]
        return c_loss
        
class select_network:
    def __init__(self, sess, num_of_feature = 4, filter_size = 3, day_length = 50, learning_rate = 1e-4, name = 'selector'):
        self.sess = sess
        self.net_name = name
        
        self._X = tf.placeholder(tf.float32,[None, day_length, num_of_feature]) # shape: batch,50,4
        self._y = tf.placeholder(tf.float32,[None])
        
        self.cell = tf.contrib.rnn.BasicLSTMCell(num_units = 5)
        self.multicell = tf.contrib.rnn.MultiRNNCell([self.cell]*2)
        self.lstm1, _states = tf.nn.dynamic_rnn(self.cell, self._X, dtype=tf.float32)
        self.value = layer.fully_connected(self.lstm1[:,-1], 1, activation_fn=None)
        
        self.loss = tf.reduce_sum(tf.square(self._y - self.value))
        
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self,s):
        self.value_hat = self.sess.run(self.value,{self._X:s})
        return self.value_hat
        
    def update(self, episode_memory):
        episode_memory = np.array(episode_memory)
        s = np.array(episode_memory[:,0].tolist())
        v = np.array(episode_memory[:,1].tolist())
        loss = 0
        for i in range(len(episode_memory)):
            loss+= self.sess.run([self.loss,self.train], {self._X: s[i], self._y: v[i]})[0]
        #print('ESM loss :',loss)
        return loss

class forecaster:
    def __init__(self, sess, num_of_feature = 4, filter_size = 3, day_length = 50, learning_rate = 1e-4, name = 'forecaster'):
        self.sess = sess
        self.net_name = name
        
        self._X = tf.placeholder(tf.float32,[None, day_length, num_of_feature])
        self._y = tf.placeholder(tf.float32,[None])
        
        self.cell = tf.contrib.rnn.BasicLSTMCell(num_units = 5)
        self.multicell = tf.contrib.rnn.MultiRNNCell([self.cell]*2)
        self.lstm1, _states = tf.nn.dynamic_rnn(self.cell, self._X, dtype=tf.float32)
        self.fc1 = layer.fully_connected(self.lstm1[:,-1], 50, activation_fn=tf.nn.leaky_relu)
        self.sigma = layer.fully_connected(self.fc1,1,activation_fn = elu)
        
        self.loss = tf.reduce_sum(2*tf.log(self.sigma)+(self._y/self.sigma)**2)
        
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self,s):
        self.sigma_hat = self.sess.run(self.sigma,{self._X:s})
        return self.sigma_hat
        
    def update(self, episode_memory):
        episode_memory = np.array(episode_memory)
        s = np.array(episode_memory[:,0].tolist())
        v = np.array(episode_memory[:,1].tolist())
        loss = 0
        for i in range(len(episode_memory)):
            loss+= self.sess.run([self.loss,self.train], {self._X: s[i], self._y: v[i]})[0]
        return loss/len(episode_memory)