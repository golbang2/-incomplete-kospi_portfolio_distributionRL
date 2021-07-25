# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:49:53 2021

@author: karig
"""

import environment
import pandas as np
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from collections import deque

class numerical:
    def __init__(self,num_of_asset,beta = 0.2, test_set = 0):
        self.m = num_of_asset
        self.env = environment.trade_env(number_of_asset = num_of_asset, train = 1, test_set = test_set)
        self.beta = beta
        self.test_set = test_set
        
        self.selected = np.random.choice(200,num_of_asset,False)
        
        self.min_time = []
        for i in self.selected:
            self.min_time.append(self.env.env_data.index_deque[i][2])
        self.min_time = np.min(self.min_time)
        
        self.close = self.env.env_data.extract_close_fixed_index(self.selected,self.min_time)
    
    def reselect(self):
        if self.test_set == 1:
            self.env = environment.trade_env(number_of_asset = self.m, train = 0, test_set = 0)
        if self.test_set == 0 :
            self.env = environment.trade_env(number_of_asset = self.m, train = 1)
            self.env.env_data.train_data_slice(self.env.env_data.test_length)
        
        self.selected = np.random.choice(200,self.m,False)
        self.min_time = []
        for i in self.selected:
            self.min_time.append(self.env.env_data.index_deque[i][2])
        self.min_time = np.min(self.min_time)
        self.close = self.env.env_data.extract_close_fixed_index(self.selected,self.min_time)
        
    def cal_y(self):
        return np.log(self.close[:,-1]/self.close[:,0])/self.min_time
    
    def y_array(self):
        deque_i = deque()
        for i in range(self.close.shape[0]):
            deque_j = deque()
            for j in range(self.close.shape[1]-2):
                deque_j.append(self.close[i,j+1]/self.close[i,j]-1)
            deque_i.append(np.array(deque_j))
        return np.array(deque_i)
                
    def cal_sigma(self):
        array = self.y_array()
        sigma = np.std(array,1)
        return sigma
    
    def cal_portfolio_sigma(self,w):
        cov = np.cov(self.y_array())
        portfolio_var = w@cov@w
        return portfolio_var
        
    def cal_portfolio_utility(self,w):
        self.portfolio_return = np.sum(self.cal_y()*w)
        self.portfolio_var = self.cal_portfolio_sigma(w)
        utility = self.portfolio_return - self.beta * self.portfolio_var
        return -utility
       
    def optimize(self):
        init_w = np.ones(self.m)/self.m
        constraints = [{'type':'eq','fun':self.constraint1}]
        result = minimize(self.cal_portfolio_utility, init_w, method='SLSQP',constraints=constraints, bounds = Bounds(0, 1))
        self.optimized_weight = result.x
        return result
    
    def constraint1(self,w):
        return np.sum(w)-1
    
    def test(self):
        self.env = environment.trade_env(number_of_asset = self.m, train = 0, test_set = self.test_set)
        self.env.reset()
        self.env.start_UBAH(self.selected,self.optimized_weight)
        value_list = []
        done = False
        while not done:
            value, done = self.env.action_UBAH(self.selected, self.optimized_weight)
            value_list.append(value)
        return np.array(value_list)/1e+8
    
class UBAH:
    def __init__(self,num_of_asset, test_set = 0):
        self.m = num_of_asset
        self.env = environment.trade_env(number_of_asset = num_of_asset, train = 0, test_set = test_set)
        
        self.selected = np.random.choice(200,num_of_asset,False)
        self.w = np.ones(self.m)/self.m
        
    def reselect(self):
        self.selected = np.random.choice(200,self.m,False)
        
    def test(self):
        self.env.reset()
        self.env.start_UBAH(self.selected,self.w)
        value_list = []
        done = False
        while not done:
            value, done = self.env.action_UBAH(self.selected, self.w)
            value_list.append(value)
        return np.array(value_list)/1e+8