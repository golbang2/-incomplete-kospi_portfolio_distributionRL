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

def price_from_yahoo(code):
    data = pdr.get_data_yahoo(code+'.KS',start = date_format('2014-01-2'),end = date_format('2021-03-20'))
    return data

ksp = np.loadtxt('d:/data/KOSPI200an.csv', delimiter=',',dtype = str)
ksp_data_lst = []

