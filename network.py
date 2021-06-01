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
    def __init__(self, sess, num_of_feature=4, day_length=50, num_of_asset=10,  filter_size = 3, learning_rate = 1e-4, regularizer_rate = 0.1, beta = 0.2, name='allocator'):
        self.sess = sess
        self.input_size=day_length
        self.output_size=num_of_asset
        self.net_name=name
        self.num_of_feature = num_of_feature
        self.filter_size = filter_size
        self.beta = beta
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        initializer = layer.xavier_initializer()
        
        self._X=tf.placeholder(tf.float32,[None,self.output_size,self.input_size,self.num_of_feature], name = "s") # shape: batch,10,50,4
        self._r=tf.placeholder(tf.float32,[None,self.output_size], name = 'r')
        self._sigma = tf.placeholder(tf.float32,[None,1,self.output_size],name = 'sigma')
        self._cov = tf.placeholder(tf.float32,[None,self.output_size,self.output_size],name = 'cov')
        
        self.conv1 = layer.conv2d(self._X, 8, [1,self.filter_size], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer)
        self.conv2 = layer.conv2d(self.conv1, 1, [1,self.input_size-self.filter_size+1], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer, weights_regularizer = regularizer)
        
        self.fc1 = layer.fully_connected(layer.flatten(self.conv2), 50, activation_fn=tf.nn.leaky_relu, weights_regularizer = regularizer)

        self.policy = layer.fully_connected(self.fc1, self.output_size, activation_fn=tf.nn.softmax)
        
        self.sigma_c = tf.matmul(self._sigma,tf.linalg.diag(self.policy))
        self.sigma_c = tf.matmul(self.sigma_c,self._cov)
        self.sigma_c = tf.matmul(self.sigma_c,tf.linalg.diag(self._sigma[:,0]))
        self.sigma_c = tf.reshape(self.sigma_c,(-1,self.output_size))
        
        self.reward = self._r - self.beta * self.sigma_c
        
        self.loss = -tf.reduce_sum(self.reward*self.policy)
        
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def predict(self, s):
        s = np.expand_dims(s,axis=0)
        self.weight=self.sess.run(self.policy, {self._X: s})
        return self.weight

    def update(self, episode_memory):
        episode_memory = np.array(episode_memory)
        s = np.array(episode_memory[:,0].tolist())
        r = np.array(episode_memory[:,1].tolist())[:,0]
        a_loss = self.sess.run([self.loss,self.train], {self._X: s, self._r: r})[0]
        return a_loss
    
class estimator:
    def __init__(self, sess, num_of_feature=4, day_length=50, num_of_asset=10,  filter_size = 3, learning_rate = 1e-4, name='allocator'):
        self.sess = sess
        self.input_size=day_length
        self.net_name=name
        self.num_of_feature = num_of_feature
        self.filter_size = filter_size
        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-3)
        initializer = layer.xavier_initializer()
        
        self._X=tf.placeholder(tf.float32,[None,num_of_asset, self.input_size,self.num_of_feature], name = "x") # shape: batch,10,50,4
        self._z = tf.placeholder(tf.float32,[None, 1, num_of_asset],name = 'z')
        
        self.conv1 = layer.conv2d(self._X, 8, [1,self.filter_size], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer)
        self.conv2 = layer.conv2d(self.conv1, 1, [1,self.input_size-self.filter_size+1], padding='VALID',activation_fn = tf.nn.leaky_relu, weights_initializer = initializer)
        
        self.fc1 = layer.fully_connected(layer.flatten(self.conv2), 50, activation_fn=tf.nn.sigmoid)
        
        self.fc_L = layer.fully_connected(self.fc1, num_of_asset * num_of_asset,activation_fn = tf.nn.tanh)
        self.L = tf.reshape(self.fc_L,(-1,num_of_asset,num_of_asset))
        self.L = (self.L + tf.transpose(self.L,perm = [0,2,1])) +1e-2
        self.lower = tf.linalg.band_part(self.L,-1,0)
        self.upper = tf.linalg.band_part(self.L,0,-1)
        
        self.fc_D = layer.fully_connected(self.fc1, num_of_asset,activation_fn=tf.nn.sigmoid)
        self.D = tf.linalg.diag(self.fc_D)*2
        
        self.expD = tf.exp(self.D)
        self.R = tf.matmul(tf.matmul(self.lower,self.expD),self.upper)
        self.upper_inv = tf.linalg.inv(self.upper)
        self.lower_inv = tf.linalg.inv(self.lower)
        self.expD_inv = tf.linalg.inv(self.expD)
        self.R_inv = tf.matmul(tf.matmul(self.upper_inv,self.expD_inv),self.lower_inv)
        
        self.multiply_inv = tf.reduce_sum(tf.matmul(tf.matmul(self._z,self.R_inv),tf.transpose(self._z,perm = [0,2,1])))
        self.logdet = tf.reduce_sum(self.D)
        
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