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

def log_likelihood(sigma, r):
    s_loss = 2 * torch.log(sigma) + (r/sigma)**2
    return 0.1 * s_loss
    
def mse(mu, r):
    m_loss = (mu-r)**2
    return m_loss

def alloc_reward(weight, individual_return, contribution_sigma, beta = 0.2):
    reward = weight * individual_return - beta * contribution_sigma
    standard_reward = reward - torch.mean(reward)
    weighted_reward = weight * standard_reward
    return weighted_reward

def elu(x):
    return (nn.ELU(1.)(x)+1 +1e-5)*0.2

def calculate_contribution(weight, individual_sigma, portfolio_sigma):
    CR = weight*individual_sigma/torch.sum(weight*individual_sigma) * portfolio_sigma
    return CR
    

class predictor(nn.Module):
    def __init__(self, s_shape, layers = 1, hidden1 = 5, hidden2 = 20, beta = 0.2, number_of_asset = 10, learning_rate = 1e-4):
        super(predictor, self).__init__()
        self.beta = beta
        self.asset_number = number_of_asset
        
        self.gru_cell = nn.GRU(s_shape[2], hidden1 , layers, batch_first = True)
        
        self.mu_fc1 = nn.Linear(hidden1,hidden2)
        self.mu_fc2 = nn.Linear(hidden2,1)

        self.sigma_fc1 = nn.Linear(hidden1,hidden2)
        self.sigma_fc2 = nn.Linear(hidden2,1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        
        self.loss_list = []
        
    def forward(self, tensor_s):
        outputs, _status = self.gru_cell(tensor_s)

        mu = self.mu_fc1(outputs[:,-1])
        mu = self.mu_fc2(F.leaky_relu(mu))
        sigma = self.sigma_fc1(outputs[:,-1])
        sigma = elu(self.sigma_fc2(F.leaky_relu(sigma)))
        
        return mu, sigma
    
    def calculate_loss(self, mu, sigma, r):
        self.v_loss = log_likelihood(sigma[:,0], r)
        self.r_loss = mse(mu[:,0], r)
        self.loss = self.v_loss + self.r_loss
        self.loss_list.append(self.loss)
    
    def optimize(self):
        loss=torch.cat(self.loss_list).sum()
        loss=loss/len(self.loss_list)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_list=[]
        
class allocator(nn.Module):
    def __init__(self, input_s, num_of_asset = 10, filter_size = 3, channel1 = 8, channel2 = 2, hidden_size = 20, learning_rate = 1e-4, regularizer_rate = 1e-2, beta = 0.2):
        super(allocator, self).__init__()
        self.day_length = input_s.shape[1]
        self.output_size = num_of_asset
        self.num_of_feature = input_s.shape[2]
        self.filter_size = filter_size
        self.beta = beta
        
        self.conv1 = nn.Conv2d(in_channels = self.num_of_feature, out_channels = channel1, kernel_size = (self.filter_size, 1))
        self.conv2 = nn.Conv2d(channel1 , channel2,(self.input_size - self.filter_size + 1, 1))
        
        self.fc = nn.Linear((channel2 * num_of_asset),hidden_size)
        self.fc_weight = nn.Linear(hidden_size, num_of_asset)
        self.fc_sigma = nn.Linear(hidden_size, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate, weight_decay = regularizer_rate)
        
        self.loss_list = []
        
    def forward(self, tensor_s):
        reshaped_s = tensor_s.view(1,4,50,10)
        x = F.leaky_relu(self.conv1(reshaped_s))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.fc(x))
        weight = self.fc_weight(x)
        sigma = self.fc_sigma(x)
        
        return weight, sigma
    
    def calculate_loss(self, weight, individual_return, portfolio_return, individual_sigma, portfolio_sigma):
        contribution_sigma = calculate_contribution(weight,individual_sigma, portfolio_sigma)
        self.s_loss = log_likelihood(portfolio_sigma, portfolio_return)
        self.w_loss = alloc_reward(weight, individual_return, contribution_sigma, self.beta)
        self.loss = self.s_loss + self.w_loss
        self.loss_list.append(self.loss)
    
    def optimize(self):
        loss=torch.cat(self.loss_list).sum()
        loss=loss/len(self.loss_list)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_list=[]
        