#!/usr/bin/env python3

import numpy as np

import tensorflow as tf
from scipy.optimize import minimize

import tensorflow_addons as tfa
import tensorflow.keras.utils as conv_utils
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import InputSpec, Concatenate
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda
tf.keras.backend.set_floatx('float64')

import matplotlib.pyplot as plt

from Models import UNet_transfer3


from observations import observations


from build_dataset import dataset

#from background import load_background

from utils_4dvar_EOF import FourDVar

from skimage.measure import block_reduce
min_sit = tf.cast(-0.4982132613658905, dtype = tf.float64)

timestep = 1
mask = np.load('../../transfer/mask.npy')
mask = block_reduce(mask, block_size = (4, 4, 1), func = np.min)
mask = 1 - mask



model =  UNet_transfer3( mask ,
                input_shape = (128, 128, 4*timestep + 6),
                kernel_size=3,
                activation = tfa.activations.mish,
                SE_prob = 0, 
                N_output = 2,
                final = 'relu',
                train = True,
                n_filter = 16,
                dropout_prob = 0)

path_to_save = '../../transfer/Results/UNet_transfer2_relu_train2/'
N_cycle = 45
time_window = 32
save_pred = '1July/EOF_obs_op_nonoise/'


inflation = 1


test = FourDVar(model, mask, k = 8871,obs_op = 'multi',starting_time = 0, std_back = 1, std_obs = 0.4,inflation = inflation, ftol = 1e-6, N_cycle = N_cycle, time_window = time_window, timestep = 1,
        save_pred = save_pred, save_grad = True, path_to_save = path_to_save, path_to_data = './')

x_4D, x_free, x_truth, x_analysis_forecast, x_free_forecast, x_analysis_forecast2 = test.test()

print(np.shape(x_free_forecast))
x_4D = x_4D.reshape((N_cycle*time_window, 128, 128))
x_free = x_free.reshape((N_cycle*time_window, 128, 128))
x_truth2 = np.array(x_truth[:N_cycle*time_window])
x_analysis_truth = np.zeros((N_cycle-2, 90, 128, 128))
x_analysis_truth2 = np.zeros((N_cycle-4, 90, 128, 128))
for i in range(N_cycle-2):
    print(i)
    x_analysis_truth[i] = x_truth[i*time_window:i*time_window + 90]
for i in range(N_cycle-4):
    x_analysis_truth2[i] = x_truth[i*time_window+ 16:i*time_window + 90 + 16]
def rmse(x, y):
    return np.sqrt(np.mean((x - y)**2, axis = (1, 2)))

def rmse_analysis(x, y):
    return np.sqrt(np.mean((x - y)**2, axis = (2, 3)))

rmse_store = rmse(x_4D*mask.squeeze(), x_truth2*mask.squeeze())
print(rmse_store)
rmse_free_store = rmse(x_free*mask.squeeze(), x_truth2*mask.squeeze())
rmse_analysis_long = rmse_analysis(x_analysis_truth*mask.squeeze(), x_analysis_forecast*mask.squeeze())
rmse_free_long = rmse_analysis(x_analysis_truth*mask.squeeze(), x_free_forecast*mask.squeeze())
rmse_analysis2 = rmse_analysis(x_analysis_truth2*mask.squeeze(), x_analysis_forecast2*mask.squeeze())
np.save(save_pred + 'rmse_analysis.npy', rmse_analysis_long)
np.save(save_pred + 'rmse_analysis_mid.npy', rmse_analysis2)
np.save(save_pred + 'rmse.npy', rmse_store)
np.save(save_pred + 'free_rmse.npy', rmse_free_store)
np.save(save_pred + 'long_free_rmse.npy', rmse_free_long)
#print(rmse_store)
#print(rmse_free_store)

np.save(save_pred + 'x_4D.npy', x_4D)
np.save(save_pred +'x_free.npy', x_free)
np.save(save_pred +'x_truth.npy', x_truth[:time_window*N_cycle])

