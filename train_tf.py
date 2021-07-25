# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:45:16 2021

@author: karig
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
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

def stan_r(r,sigma):
    dt = np.diag(sigma[:,0])
    inv_dt = np.linalg.inv(dt)
    st = inv_dt@r
    return np.expand_dims(st,0)

#preprocessed data loading
is_train = 1

#hyperparameters
input_day_size = 50
filter_size = 3
num_of_feature = 4
num_of_asset = 10
num_episodes = 3000 if is_train ==1 else 1
money = 1e+8
beta = 0.2
gamma=0.07

#saving
save_frequency = 100
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
    AAM = network.policy(sess ,num_of_asset = num_of_asset, gamma= gamma)
with tf.variable_scope('ECM'):
    ECM = network.estimator(sess,num_of_asset = num_of_asset)

sess.run(tf.global_variables_initializer())

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
    saver_ECM.restore(sess,ckpt_ECM.model_checkpoint_path)
    #saver_AAM.restore(sess,ckpt_AAM.model_checkpoint_path)

w = np.ones(num_of_asset)/num_of_asset
benchmark_sum = 0
value_list = []

for i in range(2100,num_episodes):
    exp_memory = deque()
    s = env.reset()
    s = MM_scaler(s)
    done = False
    v = money
    while not done:
        #evaluated_value = ESM.predict(s)
        forecasted_sigma = FVM.predict(s)
        #selected_s = env.select_from_value(evaluated_value-(beta * forecasted_sigma**2))
        selected_s = env.select_rand()
        selected_s = MM_scaler(selected_s)
        w = AAM.predict(selected_s)
        s_prime,r,done,v_prime,growth = env.step(w)
        s_prime = MM_scaler(s_prime)
        selected_sigma = selecting(env.index, forecasted_sigma)
        D = np.diag(selected_sigma[:,0])
        H = D@ECM.predict(selected_s)@D
        exp_memory.append([selected_s, r-r.mean(), H]) #train AAM
        #exp_memory.append([selected_s, stan_r(r,selected_sigma)]) #train ECM
        s = s_prime
        v = v_prime
        value_list.append(v)
        if done:
            print('%d agent: %.4f benchmark: %.4f' %(i,v/money,(v-env.benchmark)/money))
            benchmark_sum += (v-env.benchmark)/money
            loss = AAM.update(exp_memory)
            print(i,loss)
            
    if save_model == 1 and i % save_frequency == save_frequency - 1:
        saver_AAM.save(sess,save_path+'AAM/m'+str(num_of_asset)+'/gamma'+str(gamma)+'/AAM-'+str(i)+'.cptk')
        #saver_ECM.save(sess,save_path+'ECM/m'+str(num_of_asset)+'/ECM-'+str(i)+'.cptk')
        print(i,'saved performance', benchmark_sum)
        benchmark_sum = 0
        value_list = []

'''
class train_module(network, num_asset, beta, gamma, episodes, days = 50, filter = 3, 
                   num_feature = 4, save_frequency = 100, save_path = 'd:/data/weight/', save_model = 1):
    
    env = environment.trade_env(number_of_asset = num_asset, train = 1)
    
        
    self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    self.config.gpu_options.allow_growth = True
    
    self.sess = tf.Session(config = self.config)
    
    if network == 'ESM'
    
'''  