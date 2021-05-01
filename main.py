# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 02:14:44 2021

@author: karig
"""

import matplotlib.pyplot as plt
from collections import deque
import network_pytorch
import environment
import numpy as np
import torch

#min max scaling function
def MM_scaler(s):
    x= np.zeros(s.shape)
    for i in range(len(s)):
        x[i]=(s[i]-np.min(s[i],axis=0))/((np.max(s[i],axis=0)-np.min(s[i],axis=0))+1e-5)
    return x

#preprocessed data loading
is_train = 1

#hyperparameters
input_day_size = 50
filter_size = 3
num_of_feature = 4
num_of_asset = 10
num_episodes = 10000 if is_train ==1 else 1
money = 1e+8

use_cuda = torch.cuda.is_available()
cuda_index = torch.device('cuda:0') 


env = environment.trade_env(number_of_asset = num_of_asset)

s=env.reset()
s=MM_scaler(s)

agent = network_pytorch.Agent(s.shape)
agent = agent.cuda()

for i in range(num_episodes):
    memory = deque()
    s=env.reset()
    s=MM_scaler(s)
    done=False
    v=money
    while not done:
        mu, sigma, z = agent.predict(torch.tensor(s, dtype = torch.float).cuda())
        selected_s = env.select_rand()
        selected_s = MM_scaler(selected_s)
        
        