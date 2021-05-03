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

def softmax(z):
    softmax = torch.nn.Softmax(0)
    return softmax(z)

def selecting(z):
    softmax_z = softmax(z)
    selected = softmax_z[:,0].multinomial(num_samples = num_of_asset, replacement = False)
    selected = selected.cpu().numpy()
    xi_z = deque()
    for i in selected:
        xi_z.append(z[i])
    xi_z = torch.tensor(xi_z)
    weight = softmax(xi_z)
    return selected, weight

def utility_fn(mu,sigma):
    utility = mu-beta*sigma*sigma
    return utility

#preprocessed data loading
is_train = 1

#hyperparameters
input_day_size = 50
filter_size = 3
num_of_feature = 4
num_of_asset = 10
num_episodes = 1 if is_train ==1 else 1
money = 1e+8
beta = 0.4

use_cuda = torch.cuda.is_available()
cuda_index = torch.device('cuda:0') 


env = environment.trade_env(number_of_asset = num_of_asset)

s=env.reset()
s=MM_scaler(s)

agent = network_pytorch.Agent(s.shape, beta = beta)
agent = agent.cuda()


s=env.reset()
s=MM_scaler(s)
done=False
v=money

mu, sigma, z = agent.predict(torch.tensor(s, dtype = torch.float).cuda())
selected_index, weight = selecting(z)
selected_s = env.select_from_index(selected_index)
s_prime,r,done,v_prime,growth = env.step(weight.numpy())



'''
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
        
'''      