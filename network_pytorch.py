# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 23:09:29 2021

@author: karig
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def cal_pdf(y, mu, sigma):
    pdf = torch.exp(-0.5 * ((y - mu) / sigma)** 2) / (torch.sqrt(2 * torch.tensor(np.pi)) * sigma)
    return pdf

'''
def weighted_return(z, selected_z, mu, sigma, sensivity):
    sum_z = torch.sum(selected_z)
    w_r = (torch.exp(z) / torch.exp(sum_z)) * (mu - sensivity * (sigma**2))
    return w_r
'''

#def elu(x):
#    return torch.where(x > 0, x+1 , torch.exp(x))

def elu(x):
    return nn.ELU(1.)+1
    
import environment
env = environment.trade_env()
s = env.reset()
selected_s = env.select_rand()

class Agent(nn.Module):
    def __init__(self, s_shape, layers = 2, hidden_size = 5, beta = 0.4):
        super(Agent, self).__init__()
        self.beta = beta
        self.elu = nn.ELU(1.)
        
        self.LSTM_cell = nn.GRU(s_shape[2], hidden_size , layers, batch_first = True)
        
        self.fc_mu = nn.Linear(5,1)
        self.fc_sigma = nn.Linear(5,1)
        #self.fc_z = nn.Linear(5,1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-4)
        
        self.loss_list = []
        
    def predict(self, tensor_s):
        #tensor_s = torch.tensor(s, dtype = torch.float)
        outputs, _status = self.LSTM_cell(tensor_s)
        mu = self.fc_mu(outputs[:,-1])
        sigma = self.elu(self.fc_sigma(outputs[:,-1]))
        #z = self.fc_z(outputs[:,-1])
        
        return mu, sigma, #z
    
    def calculate_loss(self,mu,sigma,r):#(self, z, selected_z, mu, sigma, r, gamma = 0.2):
        dist_loss = cal_pdf(r, mu[:,0], sigma[:,0])
        #alloc_loss = weighted_return(z, selected_z, mu, sigma, self.beta)
        loss = - dist_loss #- gamma * alloc_loss
        self.loss_list.append(loss)
    
    def train(self):
        loss=torch.cat(self.loss_list).sum()
        loss=loss/len(self.loss_list)
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_list=[]