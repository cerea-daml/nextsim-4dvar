#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import tensorflow as tf
import xarray as xr
from tqdm import trange
from functools import partial
#import tensorflow_addons as tfa
from skimage.measure import block_reduce
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


from Models import UNet_transfer3, UNet_transfer2, UNet_transfer, ResNet, UNet3,PCNN, CNN, UNet4 , UNet4_simple

#Mask 

mask = np.load('mask.npy')
mask = block_reduce(mask, block_size=(4, 4, 1), func=np.min)
print(np.shape(mask))
mask = 1-mask

from test_adjoint import Test
timestep = 1
N_cycle = 1400
k = 30
min_sit = tf.cast(-0.4982132613658905, dtype = tf.float64)

alpha = tf.math.exp(min_sit)
beta = 1 
def my_softplus(z):
    return 1/beta*tf.math.log(beta*tf.math.exp(tf.cast(z, dtype = tf.float64)) + alpha)


model =  UNet_transfer3( mask ,
                final = 'softplus',
                train = True,
                input_shape = (128,128,4*timestep + 6),
                kernel_size=3,
                activation = tfa.activations.mish,
                SE_prob = 0, 
                N_output = 1,
                n_filter =16, 
                dropout_prob = 0)
path_to_save =  './Results/UNet_transfer2_relu_train2/'


path_to_data = '../../data/SIT_LR_full_state/'

test = Test(model = model,season = 'summer', mask = mask,k = k,  N_cycle = N_cycle, save_pred = True, timestep =timestep, path_to_save = path_to_save, path_to_data = path_to_data, noise = 0, noise_init = True)

fs, fs_pers, bias= test.test_model()
#np.save(path_to_save +'clip_cycle_'+str(N_cycle)+ '_bias_mean.npy', bias)
#np.save(path_to_save +'clip_cycle_'+str(N_cycle)+ '_fs_mean.npy', fs)
#np.save(path_to_save +'clip_cycle_'+str(N_cycle)+ '_fs_mean_pers.npy', fs_pers)
