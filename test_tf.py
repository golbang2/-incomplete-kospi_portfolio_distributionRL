# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 23:51:44 2021

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
import pandas as pd
import benchmark

def MM_scaler(s):
    x= np.zeros(s.shape)
    for i in range(len(s)):
        x[i]=(s[i]-np.min(s[i],axis=0))/((np.max(s[i],axis=0)-np.min(s[i],axis=0))+1e-5)
    return x

def APV(array):
    return array[-1]/array[0]

def get_std(apv_array):     #get_std(gamma03)
    y_array = np.log(apv_array[1:]/apv_array[:-1])
    return np.std(y_array) * np.sqrt(period)

def sharpe_ratio(value_array):
    Rf = [1.,1.008]
    sigma = get_std(value_array)
    APV = value_array[-1]
    excess_return = APV-Rf[-1]/Rf[0]
    sharpe_ratio = excess_return/sigma
    return sharpe_ratio

def MDD(apv):
    apv_max=[]
    apv_min=[]
    for i in range(1,period):
        apv_max.append(np.max(apv[:i]))
        apv_min.append(np.min(apv[np.argmax(apv[:i]):i]))
    drawdown = (np.array(apv_max) - np.array(apv_min))/np.array(apv_max)
    return np.max(drawdown)

def mean_value(array,func):
    array_deque = deque()
    for i in array:
        array_deque.append(func(i))
    return np.mean(np.array(array_deque))

#hyperparameters
input_day_size = 50
filter_size = 3
money = 1e+8
num_of_feature = 4
num_of_asset = 8
gamma = 0.1
beta = 0.2

save_path = './weights/'
load_model = 1
is_train = 0
network_result = 0
benchmark_results = 1
test_set = 0


#model
env = environment.trade_env(number_of_asset = num_of_asset, train = 0, test_set = test_set)
period = env.env_data.max_len-input_day_size

if network_result:
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    
    sess = tf.Session(config = config)
    
    with tf.variable_scope('ESM'):
        ESM = network.select_network(sess)
    with tf.variable_scope('FVM'):
        FVM = network.forecaster(sess)
    with tf.variable_scope('AAM'):
        AAM = network.policy(sess ,num_of_asset = num_of_asset, gamma = gamma)
    with tf.variable_scope('ECM'):
        ECM = network.estimator(sess,num_of_asset = num_of_asset)
    
    sess.run(tf.global_variables_initializer())
    
    saver_ESM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'ESM'))
    saver_FVM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'FVM'))
    saver_AAM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'AAM'))
    saver_ECM = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'ECM'))
    ckpt_ESM = tf.train.get_checkpoint_state(save_path+'PRM')
    ckpt_FVM = tf.train.get_checkpoint_state(save_path+'PVM')
    ckpt_AAM = tf.train.get_checkpoint_state(save_path+'AOM/m'+str(num_of_asset)+'/gamma'+str(gamma))
    ckpt_ECM = tf.train.get_checkpoint_state(save_path+'ECM/m'+str(num_of_asset))

    saver_ESM.restore(sess,ckpt_ESM.model_checkpoint_path)
    saver_FVM.restore(sess,ckpt_FVM.model_checkpoint_path)
    saver_ECM.restore(sess,ckpt_ECM.model_checkpoint_path)
    saver_AAM.restore(sess,ckpt_AAM.model_checkpoint_path)
    
    w = np.ones(num_of_asset)/num_of_asset
    value_list = []
    
    exp_memory = deque()
    s = env.reset()
    s = MM_scaler(s)
    done = False
    v = money
    while not done:
        evaluated_value = ESM.predict(s)
        forecasted_sigma = FVM.predict(s)
        selected_s = env.select_from_value(evaluated_value-(beta * forecasted_sigma**2))
        selected_s = MM_scaler(selected_s)
        w = AAM.predict(selected_s)
        s_prime,r,done,v_prime,growth = env.step(w)
        s_prime = MM_scaler(s_prime)
        s = s_prime
        v = v_prime
        value_list.append(v)
        
        if done:
            print('agent: %.4f benchmark: %.4f' %(v/money,(v-env.benchmark)/money))
            value_array = np.array(value_list)/money
            np.savetxt('./results/m'+str(num_of_asset)+'/gamma'+str(gamma)+'test'+str(test_set)+'.csv',value_array)
            plt.plot(value_array)
            
if benchmark_results:
    #kospi
    k200price = pd.read_csv("./data/kospi200_historical.csv")
    k200 = k200price[['종가']].to_numpy(dtype=np.float32)
    k200 = k200[1:-1][::-1][:,0]
    k200 = k200[test_set*175:(test_set+1)*175-1]
    k200 = k200/k200[0]
    
    #numerical
    num = benchmark.numerical(num_of_asset,beta,test_set = test_set)
    num_deque = deque()
    for i in range(100):
        num.reselect()
        num.optimize()
        numerical_value_array = num.test()
        num_deque.append(numerical_value_array)
    mean_num = np.mean(np.array(num_deque),0)
    
    #ubah
    ubah = benchmark.UBAH(num_of_asset, test_set)
    ubah_deque = deque()
    for i in range(100):
        ubah.reselect()
        ubah_deque.append(ubah.test())
    mean_ubah = np.mean(np.array(ubah_deque),0)

    '''
    mdd_num = deque()
    for i in ubah_array:
        mdd_num.append(np.max(MDD(i)))
    np.mean(np.array(mdd_num))
    '''  

    gamma03 = np.loadtxt('./results/m'+str(num_of_asset)+'/gamma'+str(0.03)+'test'+str(test_set)+'.csv')
    gamma07 = np.loadtxt('./results/m'+str(num_of_asset)+'/gamma'+str(0.07)+'test'+str(test_set)+'.csv')
    gamma10 = np.loadtxt('./results/m'+str(num_of_asset)+'/gamma'+str(0.1)+'test'+str(test_set)+'.csv')

def show_plot():    
    plt.figure(figsize = (10,4))
    plt.grid(1)
    plt.plot(gamma03,label='gamma = 0.03')
    plt.plot(gamma07,label='gamma = 0.07')
    plt.plot(gamma10,label='gamma = 0.1')
    plt.plot(k200/k200[0],label='KOSPI200')
    plt.plot(mean_ubah,label='UBAH')
    plt.plot(mean_num,label='SLSQP')
    plt.xlabel("time")
    plt.ylabel("APV")
    plt.legend()
    plt.show()
    
def PM():
    st_list = [gamma03,gamma07,gamma10,k200]
    it_list = [np.array(ubah_deque),np.array(num_deque)]
    
    for i in st_list:
        print('apv',i[-1]/i[0])
        print('mdd',MDD(i))
        print('std',get_std(i))
        print('SR', sharpe_ratio(i))
        
    for i in it_list:
        print('std',mean_value(i,get_std))
        print('mdd',mean_value(i,MDD))
        print('sr',mean_value(i,sharpe_ratio))
    print('apv',mean_ubah[-1]/mean_ubah[0])
    print('apv',mean_num[-1]/mean_num[0])
    
show_plot()
PM()