# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 23:09:29 2021

@author: karig
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

def pdf()

import environment
env = environment.trade_env()
s = env.reset()
selected_s = env.select_rand()

cell = nn.LSTM(s.shape[2], 2, num_layers = 2, batch_first=True)
    
class Agent(nn.Module):
    def __init__(self, s_shape, layers = 2, hidden_size = 5):
        super(Agent, self).__init__()
        
        self.LSTM_cell = nn.LSTM(s_shape[2], hidden_size , layers, batch_first = True)
        
        self.fc_mu = nn.Linear(5,1)
        self.fc_sigma = nn.Linear(5,1)
        self.fc_z = nn.Linear(5,1)
        
        self.optimizer = optim.Adam(self.parameters(),lr = 1e-5)
        
        self.loss_list = []
        
    def predict(self, s):
        tensor_s = torch.tensor(s)
        outputs, _status = cell(tensor_s)
        mu = self.fc_mu(outputs[:,-1])
        sigma = self.fc_sigma(outputs[:,-1])
        z = self.fc_z(outputs[:,-1])
        
        return mu, sigma, z
    
    def calculate_loss(self,s,z,sum_z):
        
    
    def stack_memory(self,loss):
        self.loss_list.append(loss)        
    
    def train(self):
        loss=torch.cat(self.loss_list).sum()
        loss=loss/len(self.loss_list)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_lst=[]