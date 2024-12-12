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

PATH_TO_DATA = '../../../data/SIT_LR_fullstate'
class observations():
    def __init__(self, PATH_TO_DATA, frequency, limit_run, std_obs, starting_time = 0):
    #Normalization constant
        self.mean_input =0.38415005912623124
        self.std_input = 0.7710474897556033
        self.mean_output = -1.6270055141928728e-05
        self.std_output = 0.023030024601018818
        self.N_x = 128
        self.N_y = 128
        self.starting_time = starting_time
        self.timestep = 1
        self.std_obs = std_obs
        self.frequency = frequency
        self.limit_run = limit_run
        self.path_data = PATH_TO_DATA
    def normalize_input(self, x):
        return (x - self.mean_input) / self.std_input

    def reverse_normalize_output(self, x):
        return  x * self.std_output + self.mean_output

    def reverse_normalize_input(self, x):
        return  x * self.std_input + self.mean_input

    def get_dataset(self, filenames, batch_size):
        dataset = self.load_dataset(filenames, self.timestep)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        return dataset

    def load_dataset(self, filenames, labeled=True):
        dataset = tf.data.TFRecordDataset(
            filenames
        )  # automatically interleaves reads from multiple files
        dataset.element_spec
        dataset = dataset.map(partial(self.read_tfrecord1))
        return dataset

    def read_tfrecord1(self, example):

        tfrecord_format = {
            "inputs": tf.io.FixedLenFeature(
                [4 * self.N_x * self.N_y * 4], tf.float32
            ),
            "outputs": tf.io.FixedLenFeature(
                [2* self.N_x * self.N_y], tf.float32
            ),
        }

        example = tf.io.parse_single_example(example, tfrecord_format)

        inputs = tf.cast(example["inputs"], tf.float32)

        inputs = tf.reshape(inputs, [*[4,  self.N_x,  self.N_x, 4]])
        inputs = tf.transpose(inputs, [1, 2, 0, 3])
        inputs_forcings = inputs[:,:,1:,-2:]
        inputs_forcings = tf.reshape(inputs_forcings, [*[ self.N_x,  self.N_x, 6]])
        inputs_past = inputs[:,:,:,1]
        inputs_past = tf.reshape(inputs_past, [*[ self.N_x,  self.N_x, 4]])
        inputs = tf.concat([inputs_past, inputs_forcings], axis = 2)
        outputs = tf.cast(example["outputs"], tf.float32)
        inputs = replacenan(inputs)
        outputs = tf.reshape(outputs, [*[ self.N_x,  self.N_x, 2]])
        return inputs, outputs
    def load_time(self):
        TIME_PATH = 'time.npy'
        #Load time information
        time = np.load(TIME_PATH, allow_pickle = True)
        #Get time for the test year
        time = time[2].indexes['time']
        #Time info for seasonal forecast
        time = time[:2*self.limit_run:2]


        return time


    def load_observation(self):
        TEST_FILENAMES = [
                self.path_data + "test.tfrecords.000",
                self.path_data + "test.tfrecords.001",
                self.path_data + "test.tfrecords.002",
                self.path_data + "test.tfrecords.003",
                self.path_data + "test.tfrecords.004",
                self.path_data + "test.tfrecords.005",
                self.path_data + "test.tfrecords.006",
                self.path_data + "test.tfrecords.007",
                self.path_data + "test.tfrecords.008",
                self.path_data + "test.tfrecords.009",
                self.path_data + "test.tfrecords.010",
                self.path_data + "test.tfrecords.011",
                self.path_data + "test.tfrecords.012",
                self.path_data + "test.tfrecords.013",
                self.path_data + "test.tfrecords.014",
                self.path_data + "test.tfrecords.015",
                self.path_data + "test.tfrecords.016",
                self.path_data + "test.tfrecords.017",
                self.path_data + "test.tfrecords.018",
                self.path_data + "test.tfrecords.019",
                self.path_data + "test.tfrecords.020",
                self.path_data + "test.tfrecords.021",
                self.path_data + "test.tfrecords.022",
                self.path_data + "test.tfrecords.023",
                self.path_data + "test.tfrecords.024",
                self.path_data + "test.tfrecords.025",
                self.path_data + "test.tfrecords.026",
                self.path_data + "test.tfrecords.027",
                self.path_data + "test.tfrecords.028",
                self.path_data + "test.tfrecords.029",
                self.path_data + "test.tfrecords.030",
                self.path_data + "test.tfrecords.031",
                self.path_data + "test.tfrecords.032",
                self.path_data + "test.tfrecords.033",
                self.path_data + "test.tfrecords.034",
                self.path_data + "test.tfrecords.035",
                self.path_data + "test.tfrecords.036",
                self.path_data + "test.tfrecords.037",
                self.path_data + "test.tfrecords.038",
                self.path_data + "test.tfrecords.039",
                self.path_data + "test.tfrecords.040",
                self.path_data + "test.tfrecords.041",
                self.path_data + "test.tfrecords.042",
                self.path_data + "test.tfrecords.043",
                self.path_data + "test.tfrecords.044",
                self.path_data + "test.tfrecords.045",
                self.path_data + "test.tfrecords.046",
                self.path_data + "test.tfrecords.047",
                self.path_data + "test.tfrecords.048"
                ]

        print(TEST_FILENAMES)
        test_dataset = self.get_dataset(TEST_FILENAMES, batch_size=2890)
        
        print(np.shape(test_dataset))
        x, y = next(iter(test_dataset))

        print(self.frequency*2)
        obs = x[self.starting_time::self.frequency*2, :, :, 0].numpy()
        
        min_sit = tf.cast(-0.4982132613658905, dtype = tf.float32)

        #np.random.seed(42)

        print(np.shape(obs))
        #obs = self.reverse_normalize_input(obs)
        #print(np.shape(obs))
        #p#rint(self.frequency)
    
        for i in range(np.shape(obs)[0]):
            #print(i)
            obs[i] = obs[i] + 0*np.random.normal(0, self.std_obs, (128, 128))

            #obs[i] = obs[i]*np.exp(np.random.normal(0, self.std_obs-self.std_obs**2, (128, 128)))
            #obs[i] = tf.clip_by_value(obs[i] + 0.4*obs[i]*np.random.normal(0, self.std_obs,(128, 128)), clip_value_min = tf.cast(0, dtype = tf.float64), clip_value_max = tf.cast(20, dtype = tf.float64))
        
        #obs = self.normalize_input(obs)
        mask = np.load('../../transfer/mask.npy')
        mask = block_reduce(mask, block_size = (4, 4, 1), func = np.min)
        obs = np.where(mask.squeeze()==True, min_sit, np.array(obs))


        time = self.load_time()
        time_obs = time[self.starting_time:self.limit_run:2*self.frequency]
        return obs, time_obs
