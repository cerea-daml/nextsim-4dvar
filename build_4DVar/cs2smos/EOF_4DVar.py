#!/usr/bin/env python3

import numpy as np
import xarray as xr
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


from Models import UNet_transfer3
from real_observations import observations
from build_dataset import dataset
from utils_4dvar_EOF import FourDVar

from skimage.measure import block_reduce

#Load and reshape mask
mask = np.load('../../data/transfer/mask.npy')
mask = block_reduce(mask, block_size = (4, 4, 1), func = np.min)
mask = 1 - mask

#Define timestep
timestep = 1

#Load truth
data = xr.open_dataset('data_obs/sit_obs_2021.nc')
x = data.analysis_sea_ice_thickness.data
x = np.nan_to_num(x, nan = -0.4982132613658905).reshape((177, 128, 128))

#Initialize model
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

#Path to model weigths
path_to_model = '../../data/transfer/Results/UNet_transfer2_relu_train2/'

#Number of cycles
N_cycle = 21

#Length of DAW
time_window = 16

#Path to save results
save_pred = '11July/test_Robs/' 

#Perform minimization
test = FourDVar(model, 
                mask, 
                k = 8871, 
                obs_op = 'multi',
                starting_time = 0, 
                std_obs = 1.2,
                inflation = 1, 
                ftol = 1e-6, 
                N_cycle = N_cycle, 
                time_window = time_window, 
                timestep = 1,
                path_to_save = save_pred, 
                save_grad = True, 
                path_to_model = path_to_model, 
                path_to_data='data_obs/')

#Retreive outputs
x_4D, x_free, x_truth, x_analysis_forecast,x_free_forecast = test.test()

#Reshape outputs
x_truth = x.reshape((177, 128, 128))
x_4D = x_4D.reshape((N_cycle*time_window, 128, 128))[::2]
x_free = x_free.reshape((N_cycle*time_window, 128, 128))[::2]

#Select daily outputs for comparison with CS2SMOS
x_analysis_forecast = x_analysis_forecast[:,::2]
x_free_forecast = x_free_forecast[:,::2]

#Create truth array to compare results
x_truth2 = np.array(x_truth[:N_cycle*time_window//2])
x_analysis_truth = np.zeros((N_cycle-1, 18, 128, 128))
for i in range(N_cycle-1):
    x_analysis_truth[i] = x_truth[i*time_window//2:i*time_window//2 + 18]

#Def rmse
def rmse(x, y):
    return np.sqrt(np.mean((x - y)**2, axis = (1, 2)))

#Def rmse
def rmse_analysis(x, y):
    return np.sqrt(np.mean((x - y)**2, axis = (2, 3)))

#Compute rmse
rmse_store = rmse(x_4D*mask.squeeze(), 
                    x_truth2*mask.squeeze())
rmse_free_store = rmse(x_free*mask.squeeze(), 
                    x_truth2*mask.squeeze())
rmse_analysis_long = rmse_analysis(x_analysis_truth*mask.squeeze(), 
                    x_analysis_forecast*mask.squeeze())
rmse_free_long = rmse_analysis(x_analysis_truth*mask.squeeze(), 
                    x_free_forecast*mask.squeeze())

#Store rmse
np.save(save_pred + 'rmse_analysis.npy', rmse_analysis_long)
np.save(save_pred + 'rmse.npy', rmse_store)
np.save(save_pred + 'free_rmse.npy', rmse_free_store)
np.save(save_pred + 'long_free_rmse.npy', rmse_free_long)

print(rmse_store)

#Print outputs
np.save(save_pred + 'x_4D.npy', x_4D)
np.save(save_pred +'x_free.npy', x_free)
np.save(save_pred + 'x_free_forecast.npy', x_free_forecast )
np.save(save_pred + 'x_analysis_forecast.npy', x_analysis_forecast )
np.save(save_pred +'x_truth.npy', x_truth[:time_window*N_cycle])
