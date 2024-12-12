#!/usr/bin/env python3


from functools import partial
import numpy as np
import numpy.ma as ma
from skimage.measure import block_reduce


import matplotlib.pyplot as plt
import datetime

import tensorflow.keras.utils as conv_utils
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import InputSpec, Concatenate
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda
tf.keras.backend.set_floatx('float64')
from Models import *
from Utils2 import *


mask = np.load('mask.npy')
print(np.shape(mask))
mask = block_reduce(mask, block_size=(4, 4, 1), func=np.min)


mask = 1-mask

path = './Results/UNet'
train_model(UNet3,  mask, path_to_save=path +'/',
        path_to_data = '../../data/SIT_LR/', 
        batch_size =16 ,kernel_size = 3,
        epochs = 500, verbose = True, 
        retrain = False,
        new_lambda = 100,
        N_output = 2,
        beta = 1,
        season = 'all',
        input_shape = (128, 128, 7),
        lambda_ = 100,
        over_compensation = True,
        mu = 0.,
        nu = 0,
        lambda_scheme = 'False',
        timestep = 1,
        learning_rate = 1e-4, dropout_prob=0,
        SE_prob = 0,
        n_filters= 32,activation = tfa.activations.mish ,loss='variance')



