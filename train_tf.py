# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:45:16 2021

@author: karig
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import network
import environment
import numpy as np

def MM_scaler(s):
    x= np.zeros(s.shape)
    for i in range(len(s)):
        x[i]=(s[i]-np.min(s[i],axis=0))/((np.max(s[i],axis=0)-np.min(s[i],axis=0))+1e-5)
    return x

def selecting(index, value):
    selected_value = []
    for i in index:
        selected_value.append(value[i])
    return np.array(selected_value)

def stan_res(r,sigma):
    dt = np.diag(sigma)
    inv_dt = np.linalg.inv(dt)
    st = inv_dt@r
    return np.expand_dims(st,0)

#preprocessed data loading
is_train = 1

#hyperparameters
input_day_size = 50
filter_size = 3
num_of_feature = 4
num_of_asset = 8
num_episodes = 1000 if is_train ==1 else 1
money = 1e+8
beta = 0.2

#saving
save_frequency = 10
save_path = 'd:/data/weight/'
save_model = 1
load_model = 1


env = environment.trade_env(number_of_asset = num_of_asset, train = is_train)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

sess = tf.Session(config = config)

with tf.variable_scope('ESM'):
    ESM = network.select_network(sess)
with tf.variable_scope('FVM'):
    FVM = network.forecaster(sess)
with tf.variable_scope('AAM'):
    AAM = network.policy(sess ,num_of_asset = num_of_asset, beta = beta)
with tf.variable_scope('ECM'):
    ECM = network.estimator(sess,num_of_asset = num_of_asset)

sess.run(tf.global_variables_initializer())

module_name = ['ESM','FVM','ECM','AAM']

saver_ESM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'ESM'))
saver_FVM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'FVM'))
saver_AAM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'AAM'))
saver_ECM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'ECM'))
ckpt_ESM = tf.train.get_checkpoint_state(save_path+'ESM')
ckpt_FVM = tf.train.get_checkpoint_state(save_path+'FVM')
ckpt_AAM = tf.train.get_checkpoint_state(save_path+'AAM/m'+str(num_of_asset))
ckpt_ECM = tf.train.get_checkpoint_state(save_path+'ECM/m'+str(num_of_asset))
if load_model:
    saver_ESM.restore(sess,ckpt_ESM.model_checkpoint_path)
    saver_FVM.restore(sess,ckpt_FVM.model_checkpoint_path)
    #saver_ECM.restore(sess,ckpt_ECM.model_checkpoint_path)
    #saver_AAM.restore(sess,ckpt_AAM.model_checkpoint_path)

w = np.ones(num_of_asset)/num_of_asset
a_loss = 0

for i in range(num_episodes):
    exp_memory = deque()
    s = env.reset()
    s = MM_scaler(s)
    done = False
    v = money
    while not done:
        #evaluated_value = ESM.predict(s)
        #forecasted_sigma = FVM.predict(s)
        selected_s = env.select_rand()
        selected_s = MM_scaler(selected_s)
        #w = AAM.predict(selected_s)
        s_prime,r,done,v_prime,growth = env.step(w)
        s_prime = MM_scaler(s_prime)
        #selected_sigma = selecting(env.random_index, forecasted_sigma)
        #covariance = ECM.predict(selected_s)
        #exp_memory.append([selected_s, r-r.mean(), covariance])
        exp_memory.append([selected_s, [r]])
        s = s_prime
        v = v_prime
        if done:
            #print('%d agent: %.4f benchmark: %.4f' %(i,v/money,(v-env.benchmark)/money))
            #a_loss = AAM.update(exp_memory)
            c_loss = ECM.update(exp_memory)
            print(i, c_loss)
            
    if save_model == 1 and i % save_frequency == save_frequency - 1:
        #saver_AAM.save(sess,save_path+'AAM/AAM-'+str(i)+'.cptk')
        #saver_ECM.save(sess,save_path+'ECM/ECM-'+str(i)+'.cptk')
        print(i,'saved')