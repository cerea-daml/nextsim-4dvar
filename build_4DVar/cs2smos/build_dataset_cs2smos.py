import numpy as np
import tensorflow as tf
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from functools import partial

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


class dataset():
    ''' Python class to build the dataset for sit thickness, atmospheric forcings and forecast'''

    def __init__(self, PATH_TO_DATA, starting_time = 0):

        '''
        Parameters:
        ------------------------
        PATH_TO_DATA: str, path to the location of tfrecords for atmospheric forcings
        starting_time: int, indice to index start of first cycle assimilation
        -----------------------
        '''

        #Normalization constant
        self.mean_input =0.38415005912623124
        self.std_input = 0.7710474897556033
        self.mean_output = -1.6270055141928728e-05
        self.std_output = 0.023030024601018818
        
        #Size of images
        self.N_x = 128
        self.N_y = 128

        #Starting time
        self.starting_time = starting_time

        #Number of timestep in emulator (by default 1, can be 2)
        self.timestep = 1

        #Path to data
        self.path_data = PATH_TO_DATA


    def normalize_input(self, x):
        '''Normalize SIT input with mean_input and std_input'''
        return (x - self.mean_input) / self.std_input

    def reverse_normalize_output(self, x):
        '''Denormalize SIT output with mean_output and std_output'''
        return  x * self.std_output + self.mean_output

    def reverse_normalize_input(self, x):
        '''Renormalize SIT input with mean_input and std_input'''
        return  x * self.std_input + self.mean_input

    def get_dataset(self, filenames, batch_size):
        '''Return tfrecords file 'filenames' as a tensorflow dataset with given batch size'''
        
        #Load dataset with filenames and correct self.timestep input
        dataset = self.load_dataset(filenames, self.timestep)

        #Select batch size
        dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset

    def get_dataset_ecmwf_forecast(self, filenames, batch_size):
        '''Return tfrecords file 'filenames' for ECMWF forecast as a tensorflow dataset with given batch size'''

        #Load dataset with filenames and correct self.timestep input
        dataset = self.load_dataset_ecmwf_forecast(filenames, self.timestep)

        #Select batch size
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        return dataset

    def load_dataset_ecmwf_forecast(self, filenames, labeled=True):
        '''load ecmwf forecast from tfrecords to dataset'''

        #Read tfrecords
        dataset = tf.data.TFRecordDataset(
            filenames
        )  # automatically interleaves reads from multiple files
        dataset.element_spec
        
        #Map tfrecords to right structure according to self.read_tfrecord2
        dataset = dataset.map(partial(self.read_tfrecord2))

        return dataset

    def load_dataset(self, filenames, labeled=True):
        '''load atmospheric forcings from tfrecords to dataset'''

        #Read tfrecords
        dataset = tf.data.TFRecordDataset(
            filenames
        )  # automatically interleaves reads from multiple files
        dataset.element_spec

        #Map tfrecords to right structure according to self.read_tfrecord
        dataset = dataset.map(partial(self.read_tfrecord1))

        return dataset


    def read_tfrecord2(self, example):

        #Indicate format of inputs [4, 512, 512, 4] and outputs [2048] (useless)
        tfrecord_format = {
            "inputs": tf.io.FixedLenFeature(
                [4 * 512 * 512 * 4], tf.float32
            ),
            "outputs": tf.io.FixedLenFeature(
                [2048], tf.float32
            ),
        }

        #Read a single example
        example = tf.io.parse_single_example(example, tfrecord_format)

        #Maps inputs in tf tensor with float32 type
        inputs = tf.cast(example["inputs"], tf.float32)

        #Reshape and transpose inputs
        inputs = tf.reshape(inputs, [4, 512, 512, 4])
        inputs = tf.transpose(inputs, [1, 2, 0, 3])
        
        #Select and flatten forcings at time t6 and t12 in inputs
        inputs_forcings = inputs[:,:,1:,-2:]
        inputs_forcings = tf.reshape(inputs_forcings, [*[ 512, 512, 6]])

        #Select and flatten sit and forcings at time t0 in inputs
        inputs_past = inputs[:,:,:,1]
        inputs_past = tf.reshape(inputs_past, [*[512, 512, 4]])

        #Concatenate inputs_past and inputs_forcings to get inputs of shape (512, 512, 10)
        inputs = tf.concat([inputs_past, inputs_forcings], axis = 2)

        #Maps outputs in tf tensor with float32 type
        outputs = tf.cast(example["outputs"], tf.float32)

        #Remove nan inputs (normally unnecessary
        inputs = replacenan(inputs)

        #Reshape output
        outputs = tf.reshape(outputs, [*[2048]])

        return inputs, outputs

    def read_tfrecord1(self, example):

        #Indicate format of inputs [5, 128, 128, 4] and outputs [1, 128, 128] (useless)
        tfrecord_format = {
            "inputs": tf.io.FixedLenFeature(
                [5 * self.N_x * self.N_y * 4], tf.float32
            ),
            "outputs": tf.io.FixedLenFeature(
                [1* self.N_x * self.N_y], tf.float32
            ),
        }
        #Read a single example
        example = tf.io.parse_single_example(example, tfrecord_format)
        
        #Maps inputs in tf tensor with float32 type
        inputs = tf.cast(example["inputs"], tf.float32)

        #Reshape and transpose inputs
        inputs = tf.reshape(inputs, [4, 5, self.N_x,  self.N_x])
        inputs = tf.transpose(inputs, [2, 3, 1, 0])
        
        #Select and flatten forcings at time t6 and t12 in inputs
        inputs_forcings = inputs[:,:,3:,1:]
        inputs_forcings = tf.reshape(inputs_forcings, [*[ self.N_x,  self.N_x, 6]])

        #Select and flatten sit and forcings at time t0 in inputs
        inputs_past = inputs[:,:,2]
        inputs_past = tf.reshape(inputs_past, [*[ self.N_x,  self.N_x, 4]])

        #Concatenate inputs_past and inputs_forcings to get inputs of shape (512, 512, 10)
        inputs = tf.concat([inputs_past, inputs_forcings], axis = 2)

         #Maps outputs in tf tensor with float32 type
        outputs = tf.cast(example["outputs"], tf.float32)

        #Remove nan inputs (normally unnecessary)
        inputs = replacenan(inputs)

        #Reshape output
        outputs = tf.reshape(outputs, [*[ self.N_x,  self.N_x, 1]])

        return inputs, outputs


    def load_time(self):

        #Path to file
        TIME_PATH = 'time.npy'

        #Load time information
        time = np.load(TIME_PATH, allow_pickle = True)

        #Get time for the test year
        time = time[2].indexes['time']

        #Time info for seasonal forecast
        time = time[::2]
        return time


    def load_data(self):
        '''Function to load dataset
        ---------------------------
        Outputs
        ---------------------------
        x_input: SIT field to access min values
        forcings, ERA5 forcings
        x_input, SIT field for first guess
        time, 
        forcings_forecast ECMWF forecast
        '''

        #Load SIT fields (observations)
        TEST_FILENAMES = [
                self.path_data + "test_2020_2021_obs.tfrecords.000",
                self.path_data + "test_2020_2021_obs.tfrecords.001",
                self.path_data + "test_2020_2021_obs.tfrecords.002",
                self.path_data + "test_2020_2021_obs.tfrecords.003",
                self.path_data + "test_2020_2021_obs.tfrecords.004",
                self.path_data + "test_2020_2021_obs.tfrecords.005"
                ]

        #Create dataset
        test_dataset = self.get_dataset(TEST_FILENAMES, batch_size=355)

        #Load ECMWF forecast
        forcings_2020_11_01 = self.get_dataset_ecmwf_forecast('forecast/ds_2020_11_01.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2020_11_09 = self.get_dataset_ecmwf_forecast('forecast/ds_2020_11_09.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2020_11_17 = self.get_dataset_ecmwf_forecast('forecast/ds_2020_11_17.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2020_11_25 = self.get_dataset_ecmwf_forecast('forecast/ds_2020_11_25.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2020_12_03 = self.get_dataset_ecmwf_forecast('forecast/ds_2020_12_03.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2020_12_11 = self.get_dataset_ecmwf_forecast('forecast/ds_2020_12_11.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2020_12_19 = self.get_dataset_ecmwf_forecast('forecast/ds_2020_12_19.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2020_12_27 = self.get_dataset_ecmwf_forecast('forecast/ds_2020_12_27.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_01_04 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_01_04.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_01_12 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_01_12.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_01_20 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_01_20.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_01_28 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_01_28.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_02_05 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_02_05.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_02_13 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_02_13.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_02_21 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_02_21.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_03_01 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_03_01.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_03_09 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_03_09.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_03_17 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_03_17.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_03_25 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_03_25.tfrecords.000', 
                                                                batch_size = 37)
        forcings_2021_04_02 = self.get_dataset_ecmwf_forecast('forecast/ds_2021_04_02.tfrecords.000', 
                                                                batch_size = 37)

        #Create array for ecmwf forecast
        forcings_forecast = np.zeros((20, 37, 128, 128, 9), dtype = np.float64)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2020_11_01))
        forcings_forecast[0] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)
        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2020_11_09))
        forcings_forecast[1] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)
        
        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2020_11_17))
        forcings_forecast[2] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2020_11_25))
        forcings_forecast[3] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2020_12_03))
        forcings_forecast[4] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2020_12_11))
        forcings_forecast[5] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2020_12_19))
        forcings_forecast[6] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2020_12_27))
        forcings_forecast[7] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_01_04))
        forcings_forecast[8] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_01_12))
        forcings_forecast[9] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_01_20))
        forcings_forecast[10] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_01_28))
        forcings_forecast[11] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_02_05))
        forcings_forecast[12] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_02_13))
        forcings_forecast[13] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_02_21))
        forcings_forecast[14] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_03_01))
        forcings_forecast[15] =  block_reduce(x_forcings[:,:,:,1:].numpy(),
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_03_09))
        forcings_forecast[16] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_03_17))
        forcings_forecast[17] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_03_25))
        forcings_forecast[18] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)

        #Coarse grain forecast and load batch
        x_forcings, _ = next(iter(forcings_2021_04_02))
        forcings_forecast[19] =  block_reduce(x_forcings[:,:,:,1:].numpy(), 
                                            block_size = (1, 4, 4, 1), func = np.mean)
        
        #Load input state
        x_input = np.load('climatology_sit_after_norm.npy')[294,:,:,0]

        #Load atmospheric forcings
        x, y = next(iter(test_dataset))

        #np.save("/cerea_raid/users/durandc/input_after2.npy", x[:50])
        
        #Load, reshape mask
        mask = np.load('../../data/transfer/mask.npy')
        mask = block_reduce(mask, block_size = (4, 4, 1), func = np.min)
        mask = 1 - mask

        #Select and forcings
        forcings = x[self.starting_time:,:,:,1:]
        forcings = tf.cast(forcings, dtype = tf.float64)
        forcings = forcings.numpy()

        #Renormalize t2m to map ERA5 statistics
        mean_t2m_forcings = tf.math.reduce_mean(forcings[:,:,:,3])
        std_t2m_forcings = tf.math.reduce_std(forcings[:,:,:,3])
        mean_t2m_forecast = tf.math.reduce_mean(forcings_forecast[:,:,:,:,2])
        std_t2m_forecast = tf.math.reduce_std(forcings_forecast[:,:,:,:,2])

        #Apply renormalization on t2m field
        forcings_forecast[:,:,:,:,2] = mean_t2m_forcings + std_t2m_forcings*(forcings_forecast[:,:,:,:,2]-mean_t2m_forecast)/std_t2m_forecast
        forcings_forecast[:,:,:,:,7] = mean_t2m_forcings + std_t2m_forcings*(forcings_forecast[:,:,:,:,7]-mean_t2m_forecast)/std_t2m_forecast
        forcings_forecast[:,:,:,:,8] = mean_t2m_forcings + std_t2m_forcings*(forcings_forecast[:,:,:,:,8]-mean_t2m_forecast)/std_t2m_forecast

        #Order to map forecast and forcings fields
        new_order = [0, 1, 2, 3, 4, 7, 5, 6, 8]
        forcings_forecast = tf.gather(forcings_forecast, new_order, axis=-1)

        #np.save("forcings_forecast.npy", forcings_forecast)
        
        #Apply mask to input
        x_input = x_input* mask.squeeze()

        #Cast forcings to tensor type
        forcings = tf.cast(forcings, dtype= tf.float64)

        #Get time info
        time = self.load_time()
        
        return x_input, forcings, x_input,time, forcings_forecast[:,::2]
