# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:52:48 2020

@author: golbang
"""

import numpy as np
import pandas as pd
from collections import deque

class load_data:
    def __init__(self,data_path = './data/',day_length=50, number_of_asset = 10, train_length = 1020, test_length = 255
                 , train = True):
        #hyper parameter
        self.number_of_asset = number_of_asset
        self.number_of_feature = 4 # close,high,low,volume
        self.train_length = train_length
        self.test_length = test_length
        self.day_length = day_length
        self.index_deque = deque()
        self.value_deque = deque()
        self.max_len = 0
        a=0

        #load kospi200 list
        self.read_csv_file = 'KOSPI200.csv'
        self.ksp_list = np.loadtxt(data_path+self.read_csv_file, delimiter=',',dtype = str)
    
        self.loaded_list = []
        self.ksp_list = self.ksp_list[:]
        
        for i in self.ksp_list:
            self.ksp_data = pd.read_csv(data_path+"stock_price/"+i[0]+".csv")
            self.ksp_data = self.ksp_data[['Close','High','Low','Volume']].to_numpy(dtype=np.float32)
            if train:
                self.ksp_data = self.ksp_data[:-self.test_length]
            else:
                self.ksp_data = self.ksp_data[-self.test_length-self.day_length:]
            self.loaded_list.append(self.ksp_data)
            self.index_deque.append([a,i[0],len(self.ksp_data),i[1]])
            if self.max_len < len(self.ksp_data):
                self.max_len = len(self.ksp_data)
            a+=1

    def extract_selected(self,index,time):
        extract_deque = deque()
        for i in index:
            extract_deque.append(self.loaded_list[i][-(self.max_len-time):-(self.max_len-time-self.day_length),:])
        return np.array(extract_deque,dtype = np.float32)

    def sampling_data(self, time):
        sample_list = []
        for i in self.index_deque:
            if i[2]-self.day_length >= self.max_len-self.day_length-time:
                sample_list.append(i[0])
        return sample_list

    def extract_close(self,index,time):
        extract_deque = deque()
        for i in index:
            extract_deque.append(self.loaded_list[i][-(self.max_len-self.day_length-time),0])
        return np.array(extract_deque,dtype = np.float32)

class trade_env:
    def __init__(self, decimal = False, day_length = 50, number_of_asset = 10, train = True, data_path = './data/'):
        self.train = train
        self.env_data = load_data(data_path = data_path, train = train,number_of_asset = number_of_asset)
        self.decimal = decimal
        self.number_of_asset = number_of_asset
        self.day_length = day_length
        
    def reset(self,money=1e+8):
        self.time = 0
        self.done = False
        self.all_index = self.env_data.sampling_data(self.time)
        self.state = self.env_data.extract_selected(self.all_index,self.time)
        self.value = money
        self.benchmark = money
        self.time_list=[]
        return self.state

    def select_from_value(self,value_array):
        self.selected_index = []
        self.sorted_value = value_array[:,0].argsort()[::-1][:self.number_of_asset]
        for i in self.sorted_value:
            self.selected_index.append(self.all_index[i])
        
        self.selected_state = self.env_data.extract_selected(self.selected_index,self.time)

        return self.selected_state
    
    def select_rand(self):
        self.random_index = np.random.choice(len(self.all_index),self.number_of_asset,False)
        self.selected_index = []
        for i in self.random_index:
            self.selected_index.append(self.all_index[i])
        self.selected_state = self.env_data.extract_selected(self.selected_index,self.time)
        return self.selected_state
    
    def select_from_index(self,selected_index):
        self.selected_index = []
        for i in selected_index:
            self.selected_index.append(self.all_index[i])
        self.selected_state = self.env_data.extract_selected(self.selected_index,self.time)
        
        return self.selected_state
        
    def holding(self,index):
        self.selected_index = index
        self.selected_state = self.env_data.extract_selected(self.selected_index,self.time)
        return self.selected_state

    def step(self,weight):
        self.benchmark_prime,_ = self.calculate_value(self.benchmark,(np.ones(self.number_of_asset,dtype = np.float32)/self.number_of_asset))
        self.value,self.r = self.calculate_value(self.value,weight)
        self.return_time = self.calculate_return_time()
        self.time += 1
        self.all_index = self.env_data.sampling_data(self.time)
        self.state_prime = self.env_data.extract_selected(self.all_index,self.time)
        self.benchmark = self.benchmark_prime
        self.time_list.append(self.done)
        if self.time == self.env_data.max_len-self.day_length-1:
            self.done = True
        
        return self.state_prime, self.r, self.done, self.value , self.return_time

    def calculate_value(self,value,weight):
        close = self.env_data.extract_close(self.selected_index,self.time-1)
        close_prime = self.env_data.extract_close(self.selected_index,self.time)
        y = (weight*value//close)
        value_prime = np.sum(y * close_prime) + np.sum(weight * value % close)
        return value_prime, np.log(close_prime/close)

    def calculate_return_time(self):
        close = self.env_data.extract_close(self.all_index,self.time-1)
        close_prime = self.env_data.extract_close(self.all_index,self.time)
        return np.log(close_prime/close)
    
    def start_UBAH(self,index,weight):
        close = self.env_data.extract_close(index,self.time)
        self.y = (weight*self.value//close)
        self.residual = np.sum(weight * self.value % close)
        self.value = np.sum(self.y*close) +self.residual
        return self.y,self.residual
    
    def action_UBAH(self,index,w):
        self.time+=1
        close = self.env_data.extract_close(index,self.time)
        self.value = np.sum(self.y * close) + self.residual
        if self.time == self.env_data.max_len-self.day_length-1:
            self.done = True
        return self.done