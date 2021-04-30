# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:40:25 2021

@author: karig
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import tensorflow_probability as tfp
#import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Layer, Flatten

def mdn_cost(mu, sigma, z, y, r, sum_z):
    dist = tfp.distributions.Normal(loc=mu, scale=sigma)
    weighted_r =  keras.backend.exp(z)/keras.backend.exp(sum_z) * r
    return tf.reduce_mean(-dist.log_prob(y) - weighted_r)

def positive_elu(x):
    return tf.nn.elu(x)+1

class predictor(Model):
    def __init__(self,s, hidden_node = 5, learning_rate = 1e-4):
        super(predictor,self).__init__()
        self.input_shape = Input(s[0].shape)
        self.y_shape = Input(shape=(1))
        self.r_shape = Input(shape=(1))
        self.sum_z_shape = Input(shape= (1))

        self.lstm = LSTM(5)(self.input_shape)
        self.mu = Dense(1)(self.lstm)
        self.sigma = Dense(1)(self.lstm)
        self.z = Dense(1)(self.lstm)

        self.loss = mdn_cost(self.mu,self.sigma, self.z, self.y_shape, self.r_shape, self.sum_z_shape)

        self.model = Model(inputs = [self.input_shape, self.y_shape], outputs = [self.mu,self.sigma])
        
        self.model.add_loss(self.loss)
        
        self.adamOptimizer = optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(optimizer= self.adamOptimizer, metrics=['mse'])
        
    def predict(self,s):
        self.pre_mu = self.mu_model.predict(s)
        self.pre_sigma = self.sigma_model.predict(s)
        return self.pre_mu, self.pre_sigma
    
    def update(self,s,z,y,r,sum_z):
        return self.model.fit([s, z, y, r, sum_z], verbose=0, epochs=1, batch_size = s.shape[0])
        
import environment
env = environment.trade_env()
s = env.reset()
selected_s = env.select_rand()