# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 02:14:44 2021

@author: karig
"""

import network_pytorch
import environment
import numpy as np
import torch
import os

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
p_loss = 0.
a_loss = 0.
excess_return = 0.

predictor = network_pytorch.predictor(s.shape).to(cuda_index)
allocator = network_pytorch.allocator(s, num_of_asset, beta = beta).to(cuda_index)
predictor = predictor.cuda()
allocator = allocator.cuda()

if load_weight:
    checkpoint = torch.load(save_path+load_list[-1])
    predictor.load_state_dict(checkpoint['predictor_state_dict'])
    allocator.load_state_dict(checkpoint['allocator_state_dict'])
    predictor.optimizer.load_state_dict(checkpoint['optimizerP_state_dict'])
    allocator.optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])
    #agent = torch.load(save_path+load_list[-1])
    print(load_list[-1], 'loaded')
    iteration = int(load_list[-1][:4])
    predictor.eval()
    allocator.eval()
    if is_train:
        predictor.train()
        allocator.train()


for i in range(iteration,iteration + num_episodes):
    s=env.reset()
    s=MM_scaler(s)
    done=False
    v=money
    while not done:
        mu, sigma = predictor.forward(torch.tensor(s, dtype = torch.float).cuda())
        selected_s = env.select_rand()
        selected_s = MM_scaler(selected_s)
        weight, sigma_p = allocator.forward(torch.tensor(selected_s, dtype = torch.float).cuda())
        s_prime, r, done, v_prime, growth = env.step(weight.detach().cpu().numpy())
        s_prime = MM_scaler(s_prime)
        selected_sigma = selecting(env.random_index, sigma)
        predictor.calculate_loss(mu,sigma,torch.tensor(growth).cuda())
        allocator.calculate_loss(weight,torch.tensor(r).cuda(),torch.log(torch.tensor(v_prime/v)), selected_sigma, sigma_p)
        s = s_prime
        v = v_prime
        if done:
            excess_return += (v-env.benchmark)/money
            p_loss += torch.sum(predictor.loss).item()
            a_loss += torch.sum(allocator.loss).item()
            predictor.optimize()
            allocator.optimize()
            print('%d agent: %.4f benchmark: %.4f'
                  %(i,v/money,(v-env.benchmark)/money))
            print('mu_loss: %.4f sigma_loss: %.4f alloc_loss: %.4f sigmaP_loss: %.4f'
                  %(torch.sum(predictor.r_loss).item(),torch.sum(predictor.v_loss).item(),
                    torch.sum(allocator.w_loss).item(),torch.sum(allocator.s_loss).item()))

    if i % save_frequency == 0 and is_save == True and i !=iteration:
        torch.save({
            'predictor_state_dict': predictor.state_dict(),
            'allocator_state_dict': allocator.state_dict(),
            'optimizerP_state_dict': predictor.optimizer.state_dict(),
            'optimizerA_state_dict': allocator.optimizer.state_dict(),
            }, save_path + str(i).zfill(4)+'.tar')
        print(i, 'saved excess: %.4f Ploss: %.4f Aloss: %.4f'%(excess_return,p_loss,a_loss))
        a_loss = 0.
        p_loss = 0.
        excess_return = 0.