# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:35:26 2021

@author: karig
"""

import numpy as np
from urllib.request import urlopen
import bs4
import datetime as dt
import pandas as pd
from collections import deque
import time
import pandas_datareader as pdr

def date_format(date):
    date=str(date).replace('-','.')
    yyyy=int(date.split('.')[0])
    mm=int(date.split('.')[1])
    dd=int(date.split('.')[2])
    date=dt.date(yyyy,mm,dd)
    return date

def price_from_yahoo(code,start = '2019-8-27', end = '2021-04-27'):
    data = pdr.get_data_yahoo(code+'.KS',start = date_format(start),end = date_format(end))
    return data

'''
data_path = './data/test/'
ksp = np.loadtxt('d:/data/KOSPI200an.csv', delimiter=',',dtype = str)
for i in ksp[-60:]:
    data = price_from_yahoo(i[0])
    data.to_csv(data_path + i[0]+'.csv')
    time.sleep(2)
'''