import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import environment

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import tensorflow_probability as tfp
#import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Layer, Flatten

#min max scaling function
def MM_scaler(s):
    x= np.zeros(s.shape)
    for i in range(len(s)):
        x[i]=(s[i]-np.min(s[i],axis=0))/((np.max(s[i],axis=0)-np.min(s[i],axis=0))+1e-5)
    return x

def mdn_cost(mu, sigma, z, y, r, sum_z):
    dist = tfp.distributions.Normal(loc=mu, scale=sigma)
    weighted_r =  keras.backend.exp(z)/keras.backend.exp(sum_z) * r
    return tf.reduce_mean(-dist.log_prob(y)-weighted_r)

#preprocessed data loading
is_train = 1

#hyperparameters
input_day_size = 50
filter_size = 3
num_of_feature = 4
num_of_asset = 10
num_episodes = 10000 if is_train ==1 else 1
money = 1e+8
learning_rate = 1e-5

env = environment.trade_env(number_of_asset = num_of_asset)

s=env.reset()
s=MM_scaler(s)

input_shape = Input(s[0].shape)
y_shape = Input(shape=(1))
r_shape = Input(shape=(1))
sum_z_shape = Input(shape= (1))

lstm = LSTM(5)(input_shape)
mu = Dense(1)(lstm)
sigma = Dense(1)(lstm)
z = Dense(1)(lstm)

loss = mdn_cost(mu, sigma, z, y_shape, r_shape, sum_z_shape)

model = Model(inputs = [input_shape, y_shape, r_shape], outputs = [mu,sigma,z])

model.add_loss(loss)

adamOptimizer = optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer= adamOptimizer, metrics=['mse'])

w = np.random.rand(20)*5
softmax_w = softmax(w)

x = range(1,21)
plt.bar(x,softmax_w)
plt.xticks(range(1,21))
plt.xlabel("index number")
plt.ylabel("softmax z distribution")
plt.show()

selected = np.random.choice(20, 5, replace = 0, p = softmax_w)
selected_index = []
portfolio = []
for i in selected:
    portfolio.append(w[i])
    selected_index.append(str(i))
    
portfolio = np.array(portfolio)
softmax_p = softmax(portfolio)

plt.bar(selected_index, softmax_p)
plt.xlabel("index number")
plt.ylabel("portfolio weight")
plt.show()