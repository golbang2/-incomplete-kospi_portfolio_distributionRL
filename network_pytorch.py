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

def weighted_return(z, selected_z, mu, sigma, sensivity = 0.4):
    sum_z = torch.sum(selected_z)
    w_r = (torch.exp(z) / torch.exp(sum_z)) * (mu - sensivity * (sigma**2))
    return w_r[:,0]

def elu(x):
    return nn.ELU(1.)(x)+1

class Agent(nn.Module):
    def __init__(self, s_shape, layers = 2, hidden_size = 5, beta = 0.4):
        super(Agent, self).__init__()
        self.beta = beta
        
        self.LSTM_cell = nn.GRU(s_shape[2], hidden_size , layers, batch_first = True)
        
        self.fc_mu = nn.Linear(5,1)
        self.fc_sigma = nn.Linear(5,1)
        self.fc_z = nn.Linear(5,1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-4)
        
        self.loss_list = []
        
    def predict(self, tensor_s):
        #tensor_s = torch.tensor(s, dtype = torch.float)
        outputs, _status = self.LSTM_cell(tensor_s)
        mu = self.fc_mu(outputs[:,-1])
        sigma = elu(self.fc_sigma(outputs[:,-1]))
        z = self.fc_z(outputs[:,-1])
        
        return mu, sigma, z
    
    def calculate_loss(self, z, selected_z, mu, sigma, r):
        self.dist_loss = cal_pdf(r, mu[:,0], sigma[:,0])
        self.alloc_loss = weighted_return(z, selected_z, mu, sigma)
        self.loss = - self.dist_loss - self.alloc_loss
        self.loss_list.append(self.loss)
    
    def optimize(self):
        loss=torch.cat(self.loss_list).sum()
        loss=loss/len(self.loss_list)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_list=[]