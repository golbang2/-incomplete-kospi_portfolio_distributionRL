# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 02:14:44 2021

@author: karig
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import environment
import numpy as np
import os
import train_tf
import network

#min max scaling function
def MM_scaler(s):
    x= np.zeros(s.shape)
    for i in range(len(s)):
        x[i]=(s[i]-np.min(s[i],axis=0))/((np.max(s[i],axis=0)-np.min(s[i],axis=0))+1e-5)
    return x

def selecting(index, value):
    selected_value = []
    for i in index:
        selected_value.append(value[i])
    return torch.tensor(selected_value).cuda()

#preprocessed data loading
is_train = True
is_save = True
load_weight = 1

#hyperparameters
input_day_size = 50
num_of_feature = 4
num_of_asset = 10
num_episodes = 0 if is_train ==1 else 0
num_episodes += 1
money = 1e+8
beta = 0.2

#model
save_frequency = 10

use_cuda = torch.cuda.is_available()
cuda_index = torch.device('cuda:0') 
save_path = 'd:/data/weights/'
load_list = os.listdir(save_path)

env = environment.trade_env(number_of_asset = num_of_asset, train = is_train)

s=env.reset()
iteration = 0
excess_return = 0.

#preprocessed data loading
is_train = 1

#hyperparameters
input_day_size = 50
filter_size = 3
num_of_feature = 4
num_of_asset = 10
num_episodes = 5000 if is_train ==1 else 1
money = 1e+8
beta = 0.03

#saving
save_frequency = 100
save_path = 'd:/data/weight/'
save_model = 1
load_model = 1

