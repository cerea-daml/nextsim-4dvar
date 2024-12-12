import numpy as np
import tensorflow as tf
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from functools import partial
import xarray as xr
def replacenan(t):
    '''
    replace nan value in a tensorflow tensor
    Parameters :
    -------------------
    t : tensorflow array

    Outputs :
    --------------------
    tensor array of the same shape
    '''
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

class observations():
    '''Python class to access observations for 4D-Var minimization'''

    def __init__(self, std_obs, starting_time, frequency):
        '''
        --------------------
        Parameters:
        --------------------
        std_obs: int, sigma_obs for 4D-Var
        starting_time: int, start of first cycle
        '''
        
        #Normalization constant
        self.mean_input =0.38415005912623124
        self.std_input = 0.7710474897556033
        self.mean_output = -1.6270055141928728e-05
        self.std_output = 0.023030024601018818
        
        #Shape of fields
        self.N_x = 128
        self.N_y = 128

        #Starting time
        self.starting_time = starting_time

        #Number of timestep in the emulator inputs
        self.timestep = 1

        #Observation error
        self.std_obs = std_obs

        #Frequency of observations
        self.frequency = frequency

    def normalize_input(self, x):
        return (x - self.mean_input) / self.std_input

    def reverse_normalize_output(self, x):
        return  x * self.std_output + self.mean_output

    def reverse_normalize_input(self, x):
        return  x * self.std_input + self.mean_input


    def load_observation(self):
        '''function to return observations
        --------------------------
        Outputs
        --------------------------
        obs, np array with observations every <frequency>
        '''

        #Load CS2SMOS observations and select SIT
        data = xr.open_dataset('data_obs/sit_obs_2021.nc')
        x = data.analysis_sea_ice_thickness.data

        #Convert nan to minimum SIT value
        x = np.nan_to_num(x, nan = -0.4982132613658905)

        #Select frequency
        obs = x[::self.frequency//2, 0] 

        #Get min 
        min_sit = tf.cast(-0.4982132613658905, dtype = tf.float64)

        #Load mask
        mask = np.load('../../data/transfer/mask.npy')
        mask = block_reduce(mask, block_size = (4, 4, 1), func = np.min)
        
        
        #obs = np.where(mask.squeeze()==True, min_sit, np.array(obs))

        return obs
