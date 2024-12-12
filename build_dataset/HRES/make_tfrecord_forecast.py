import numpy as np
from tqdm import trange
import numpy.ma as ma
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import xarray as xr
def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.
    
    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.
    
    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.
    
    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:  
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))
            
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank, 
                               # it should be rshape before fed to this function.
    assert isinstance(Y, np.ndarray) or Y is None
    
    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)
    
    assert X.shape[0] == Y.shape[0]
    assert len(Y.shape) == 2
    dtype_feature_y = _dtype_feature(Y)            
    
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in trange(X.shape[0]):
        x = X[idx]
        
        y = Y[idx]
        
        d_feature = {}
        d_feature['inputs'] = dtype_feature_x(x)
        
        d_feature['outputs'] = dtype_feature_y(y)
            
        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))

def split_tfrecord(tfrecord_path, split_size):
    with tf.Graph().as_default(), tf.Session() as sess:
        ds = tf.data.TFRecordDataset(tfrecord_path).batch(split_size)
        batch = ds.make_one_shot_iterator().get_next()
        part_num = 0
        while True:
            try:
                records = sess.run(batch)
                part_path = tfrecord_path + '.{:03d}'.format(part_num)
                with tf.python_io.TFRecordWriter(part_path) as writer:
                    for record in records:
                        writer.write(record)
                part_num += 1
            except tf.errors.OutOfRangeError: break
        
#################################    
##      Test and Use Cases     ##
#################################

# 1-1. Saving a dataset with input and label (supervised learning)
data = xr.open_dataset('ds_2021_01_04_input.nc')

data = data.drop_dims(['x','y'])
data = data.to_array().to_numpy()
print(np.shape(data))
X = data
X = np.swapaxes(X,0, 1)
X = X.reshape((np.shape(X)[0],-1))

Y = data[:,:,0]
Y = np.swapaxes(Y,0, 1)
Y = Y[:,0]
Y = Y.reshape((np.shape(Y)[0],-1))
print(np.shape(Y))
np_to_tfrecords(X, Y, 'ds_2021_01_04', verbose=True)
split_tfrecord('ds_2021_01_04.tfrecords', 60)

data = xr.open_dataset('ds_2020_11_01_input.nc')

data = data.drop_dims(['x','y'])
data = data.to_array().to_numpy()
print(np.shape(data))
X = data
X = np.swapaxes(X,0, 1)
X = X.reshape((np.shape(X)[0],-1))

Y = data[:,:,0]
Y = np.swapaxes(Y,0, 1)
Y = Y[:,0]
Y = Y.reshape((np.shape(Y)[0],-1))
print(np.shape(Y))
np_to_tfrecords(X, Y, 'ds_2020_11_01', verbose=True)
split_tfrecord('ds_2020_11_01.tfrecords', 60)


data = xr.open_dataset('ds_2020_11_09_input.nc')

data = data.drop_dims(['x','y'])
data = data.to_array().to_numpy()
print(np.shape(data))
X = data
X = np.swapaxes(X,0, 1)
X = X.reshape((np.shape(X)[0],-1))

Y = data[:,:,0]
Y = np.swapaxes(Y,0, 1)
Y = Y[:,0]
Y = Y.reshape((np.shape(Y)[0],-1))
print(np.shape(Y))
np_to_tfrecords(X, Y, 'ds_2020_11_09', verbose=True)
split_tfrecord('ds_2020_11_09.tfrecords', 60)


data = xr.open_dataset('ds_2020_11_17_input.nc')

data = data.drop_dims(['x','y'])
data = data.to_array().to_numpy()
print(np.shape(data))
X = data
X = np.swapaxes(X,0, 1)
X = X.reshape((np.shape(X)[0],-1))

Y = data[:,:,0]
Y = np.swapaxes(Y,0, 1)
Y = Y[:,0]
Y = Y.reshape((np.shape(Y)[0],-1))
print(np.shape(Y))
np_to_tfrecords(X, Y, 'ds_2020_11_17', verbose=True)
split_tfrecord('ds_2020_11_17.tfrecords', 60)


data = xr.open_dataset('ds_2020_11_25_input.nc')

data = data.drop_dims(['x','y'])
data = data.to_array().to_numpy()
print(np.shape(data))
X = data
X = np.swapaxes(X,0, 1)
X = X.reshape((np.shape(X)[0],-1))

Y = data[:,:,0]
Y = np.swapaxes(Y,0, 1)
Y = Y[:,0]
Y = Y.reshape((np.shape(Y)[0],-1))
print(np.shape(Y))
np_to_tfrecords(X, Y, 'ds_2020_11_25', verbose=True)
split_tfrecord('ds_2020_11_25.tfrecords', 60)


data = xr.open_dataset('ds_2020_12_03_input.nc')

data = data.drop_dims(['x','y'])
data = data.to_array().to_numpy()
print(np.shape(data))
X = data
X = np.swapaxes(X,0, 1)
X = X.reshape((np.shape(X)[0],-1))

Y = data[:,:,0]
Y = np.swapaxes(Y,0, 1)
Y = Y[:,0]
Y = Y.reshape((np.shape(Y)[0],-1))
print(np.shape(Y))
np_to_tfrecords(X, Y, 'ds_2020_12_03', verbose=True)
split_tfrecord('ds_2020_12_03.tfrecords', 60)


data = xr.open_dataset('ds_2020_12_11_input.nc')

data = data.drop_dims(['x','y'])
data = data.to_array().to_numpy()
print(np.shape(data))
X = data
X = np.swapaxes(X,0, 1)
X = X.reshape((np.shape(X)[0],-1))

Y = data[:,:,0]
Y = np.swapaxes(Y,0, 1)
Y = Y[:,0]
Y = Y.reshape((np.shape(Y)[0],-1))
print(np.shape(Y))
np_to_tfrecords(X, Y, 'ds_2020_12_11', verbose=True)
split_tfrecord('ds_2020_12_11.tfrecords', 60)


data = xr.open_dataset('ds_2020_12_19_input.nc')

data = data.drop_dims(['x','y'])
data = data.to_array().to_numpy()
print(np.shape(data))
X = data
X = np.swapaxes(X,0, 1)
X = X.reshape((np.shape(X)[0],-1))

Y = data[:,:,0]
Y = np.swapaxes(Y,0, 1)
Y = Y[:,0]
Y = Y.reshape((np.shape(Y)[0],-1))
print(np.shape(Y))
np_to_tfrecords(X, Y, 'ds_2020_12_19', verbose=True)
split_tfrecord('ds_2020_12_19.tfrecords', 60)


data = xr.open_dataset('ds_2020_12_27_input.nc')

data = data.drop_dims(['x','y'])
data = data.to_array().to_numpy()
print(np.shape(data))
X = data
X = np.swapaxes(X,0, 1)
X = X.reshape((np.shape(X)[0],-1))

Y = data[:,:,0]
Y = np.swapaxes(Y,0, 1)
Y = Y[:,0]
Y = Y.reshape((np.shape(Y)[0],-1))
print(np.shape(Y))
np_to_tfrecords(X, Y, 'ds_2020_12_27', verbose=True)
split_tfrecord('ds_2020_12_27.tfrecords', 60)


#data = xr.open_dataset('ds_2021_03_25_input.nc')

#data = data.drop_dims(['x','y'])
#data = data.to_array().to_numpy()
#print(np.shape(data))
#X = data
#X = np.swapaxes(X,0, 1)
#X = X.reshape((np.shape(X)[0],-1))

#Y = data[:,:,0]
#Y = np.swapaxes(Y,0, 1)
#Y = Y[:,0]
#Y = Y.reshape((np.shape(Y)[0],-1))
#print(np.shape(Y))
#np_to_tfrecords(X, Y, 'ds_2021_03_25', verbose=True)
#split_tfrecord('ds_2021_03_25.tfrecords', 60)


#data = xr.open_dataset('ds_2021_04_02_input.nc')

#data = data.drop_dims(['x','y'])
#data = data.to_array().to_numpy()
#print(np.shape(data))
#X = data
#X = np.swapaxes(X,0, 1)
#X = X.reshape((np.shape(X)[0],-1))

#Y = data[:,:,0]
#Y = np.swapaxes(Y,0, 1)
#Y = Y[:,0]
#Y = Y.reshape((np.shape(Y)[0],-1))
#print(np.shape(Y))
#np_to_tfrecords(X, Y, 'ds_2021_04_02', verbose=True)
#split_tfrecord('ds_2021_04_02.tfrecords', 60)

