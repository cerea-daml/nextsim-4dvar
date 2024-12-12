import numpy as np
import scipy.stats as stats

import tensorflow_addons as tfa
import numpy.ma as ma
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from functools import partial
def load_dataset(filenames, timestep, labeled=True):
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset.element_spec
    if timestep == 4 : 
        dataset = dataset.map(
            partial(read_tfrecord4))
    elif timestep == 3 : 
        dataset = dataset.map(
            partial(read_tfrecord3))
    elif timestep == 2 : 
        dataset = dataset.map(
            partial(read_tfrecord2))
    elif timestep == 1 : 
        dataset = dataset.map(
            partial(read_tfrecord1))
    return dataset

def read_tfrecord1(example):

    tfrecord_format = (
        {
            "inputs": tf.io.FixedLenFeature([4*128*128*4],tf.float32),
            "outputs": tf.io.FixedLenFeature([2*128*128],tf.float32),
        })

    example = tf.io.parse_single_example(example, tfrecord_format)

    inputs = tf.cast(example["inputs"], tf.float32)

    inputs = tf.reshape(inputs, [*[4, 128, 128, 4]])
    inputs = tf.transpose(inputs, [1, 2, 0, 3])
    inputs_forcings = inputs[:,:,1:,-1:]
    inputs_forcings = tf.reshape(inputs_forcings, [*[128, 128, 3]])
    inputs_past = inputs[:,:,:,1]
    inputs_past = tf.reshape(inputs_past, [*[128, 128, 4]])
    inputs = tf.concat([inputs_past, inputs_forcings], axis = 2)
    outputs = tf.cast(example["outputs"], tf.float32)
    inputs = replacenan(inputs)
    outputs = tf.reshape(outputs, [*[2, 128, 128]])
    #outputs = outputs[0]
    #inputs = inputs + tf.random.normal((128, 128, 10), 0, 0.0)
    outputs = tf.reshape(outputs, [*[128, 128, 2]])   
    #outputs = outputs[:,:,0]
    return inputs, outputs


def read_tfrecord2(example):
    
    tfrecord_format = (
        {
            "inputs": tf.io.FixedLenFeature([4*512*512*4],tf.float32),
            "outputs": tf.io.FixedLenFeature([2*512*512],tf.float32),
        })

    example = tf.io.parse_single_example(example, tfrecord_format)
    
    inputs = tf.cast(example["inputs"], tf.float32)

    inputs = tf.reshape(inputs, [*[4, 512, 512, 4]])
    inputs = tf.transpose(inputs, [1, 2, 0, 3])
    inputs_forcings = inputs[:,:,1:,-2:]
    inputs_forcings = tf.reshape(inputs_forcings, [*[512, 512, 6]])
    inputs_past = inputs[:,:,:,:2]
    inputs_past = tf.reshape(inputs_past, [*[512, 512, 4 * 2]])
    inputs = tf.concat([inputs_past, inputs_forcings], axis = 2)
    outputs = tf.cast(example["outputs"], tf.float32)
    inputs = replacenan(inputs)
    outputs = tf.reshape(outputs, [*[512, 512, 2]])

    return inputs, outputs
def read_tfrecord3(example):

    tfrecord_format = (
        {
            "inputs": tf.io.FixedLenFeature([5*512*512*6],tf.float32),
            "outputs": tf.io.FixedLenFeature([2*512*512],tf.float32),
        })

    example = tf.io.parse_single_example(example, tfrecord_format)

    inputs = tf.cast(example["inputs"], tf.float32)

    inputs = tf.reshape(inputs, [*[5, 512, 512, 6]])
    inputs = tf.transpose(inputs, [1, 2, 0, 3])
    inputs_forcings = inputs[:,:,1:,-2:]
    inputs_forcings = tf.reshape(inputs_forcings, [*[512, 512, 8]])
    inputs_past = inputs[:,:,:,1:4]
    inputs_past = tf.reshape(inputs_past, [*[512, 512, 5 * 3]])
    inputs = tf.concat([inputs_past, inputs_forcings], axis = 2)
    outputs = tf.cast(example["outputs"], tf.float32)
    inputs = replacenan(inputs)
    outputs = tf.reshape(outputs, [*[512, 512, 2]])
    
    return inputs, outputs

def read_tfrecord4(example):

    tfrecord_format = (
        {
            "inputs": tf.io.FixedLenFeature([5*512*512*6],tf.float32),
            "outputs": tf.io.FixedLenFeature([2*512*512],tf.float32),
        })

    example = tf.io.parse_single_example(example, tfrecord_format)
    inputs = tf.cast(example["inputs"], tf.float32)

    inputs = tf.reshape(inputs, [*[5, 512, 512, 6]])
    inputs = tf.transpose(inputs, [1, 2, 0, 3])
    inputs_forcings = inputs[:,:,1:,-2:]
    inputs_forcings = tf.reshape(inputs_forcings, [*[512, 512, 8]])
    inputs_past = inputs[:,:,:,:4]
    inputs_past = tf.reshape(inputs_past, [*[512, 512, 5 * 4]])
    inputs = tf.concat([inputs_past, inputs_forcings], axis = 2)
    outputs = tf.cast(example["outputs"], tf.float32)
    inputs = replacenan(inputs)
    outputs = tf.reshape(outputs, [*[512, 512, 2]])
    return inputs, outputs


def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)



def get_dataset(timestep,filenames,batch_size):
    print('load dataset')
    dataset = load_dataset(filenames, timestep)
    dataset = dataset.shuffle(buffer_size = batch_size)
    dataset = dataset.prefetch(buffer_size = 2*batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
     
    return dataset

class LRFind(tf.keras.callbacks.Callback): 
    def __init__(self, min_lr, max_lr, n_rounds): 
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_up = (max_lr / min_lr) ** (1 / n_rounds)
        self.lrs = []
        self.losses = []
                                                         
        def on_train_begin(self, logs=None):
            self.weights = self.model.get_weights()
            self.model.optimizer.lr = self.min_lr

        def on_train_batch_end(self, batch, logs=None):
            self.lrs.append(self.model.optimizer.lr.numpy())
            print(self.model.optimizer.lr.numpy())
            self.losses.append(logs["loss"])
            print(logs["loss"])
            self.model.optimizer.lr = self.model.optimizer.lr * self.step_up
            if self.model.optimizer.lr > self.max_lr:
                self.model.stop_training = True
        def on_train_end(self, logs=None):
            self.model.set_weights(self.weights)

def create_power_spectra(image, N_pix) :
    npix = N_pix
    N = np.shape(image)[0]
    abins = np.zeros((N, N_pix//2))
    for n in range(N):
        fourier_image = np.fft.fftn(image[n,:,:,0] )
        fourier_amplitudes = np.abs(fourier_image)**2
        kfreq = np.fft.fftfreq(npix) * npix

        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        kbins = np.arange(0.5, npix//2+1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])


        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    abins[n] = Abins
    return abins.mean(axis = 0)

def train_model(model,mask, batch_size,  input_shape,lambda_,mu,
        path_to_save, path_to_data,loss,nu,
        n_filters, timestep,
        over_compensation = True, 
        N_output = 2,
        beta = 1,
        season = 'winter',
        lambda_scheme = 'step',
        retrain = False,
        new_lambda = 0,
        weight_init = 5e-4, 
        SE_prob = 0.3,
        kernel_size = 3, activation = 'relu', dropout_prob = 0.2, epochs = 1000, verbose = False, learning_rate = 1e-4) :

    GCS_PATH = path_to_data
    
    #Select int between 1 and 6
    
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    years = np.array(['2009','2010','2011','2012','2013','2014','2015','2016'])

    if season == 'all':
        TRAINING_FILENAMES = tf.io.gfile.glob(path_to_data + "train_20*")
        VALID_FILENAMES = tf.io.gfile.glob(path_to_data + "val*")
        TEST_FILENAMES = tf.io.gfile.glob(path_to_data + "test*")
    if season == 'winter':
        TRAINING_FILENAMES = tf.io.gfile.glob(path_to_data + "train_winter_20*")
        VALID_FILENAMES = tf.io.gfile.glob(path_to_data + "val_winter*")
        TEST_FILENAMES = tf.io.gfile.glob(path_to_data + "test_winter*")
    if season=='summer':
        TRAINING_FILENAMES = tf.io.gfile.glob(path_to_data + "train_summer_20*")
        VALID_FILENAMES = tf.io.gfile.glob(path_to_data + "val_summer*")
        TEST_FILENAMES = tf.io.gfile.glob(path_to_data + "test_summer*")

    train_dataset = get_dataset(timestep,TRAINING_FILENAMES,batch_size)
    valid_dataset = get_dataset(timestep, VALID_FILENAMES, batch_size)
    test_dataset = get_dataset(timestep, TEST_FILENAMES,batch_size)
    tf.config.run_functions_eagerly(True)
    print('### Define Model ###')
    
    min_sit = tf.cast(-0.4982132613658905, dtype = tf.float64)

    alpha = tf.math.exp(min_sit)
    beta = beta
    def my_softplus(z):
        return 1/beta*tf.math.log(beta*tf.math.exp(tf.cast(z, dtype = tf.float64)) + alpha)



    model = model(mask=mask, input_shape = input_shape, 
                            activation_final_layer = my_softplus,
                            kernel_size = kernel_size,
                            over_compensation = over_compensation,
                            activation = activation,
                            depth = 3,
                            N_output = N_output,
                            SE_prob = SE_prob,
                            n_filter = n_filters,
                            dropout_prob = dropout_prob)
        
    #Define optimizer with its learning rate
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
                [10000, 20000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = learning_rate * schedule(step)
    wd = lambda: 1e-6 * schedule(step)
    lambda2 = lambda_

    if lambda_scheme == 'step':
        boundaries = np.arange(1000, 20000, 1000)
        values = np.logspace(0, 2, 20)
        schedule_lambda = tf.optimizers.schedules.PiecewiseConstantDecay(
                [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                    11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000],
                [  1.,1.27427499,1.62377674 ,2.06913808 ,2.6366509,
                3.35981829,   4.2813324,    5.45559478,   6.95192796,   8.8586679,
                11.28837892,  14.38449888,  18.32980711,  23.35721469,  29.76351442,
                37.92690191,  48.32930239,  61.58482111,  78.47599704, 100.        ])
        step_lambda = tf.Variable(0, trainable=False)
        lambda2 = 1 * schedule_lambda(step_lambda)
    else :
        lambda2 = lambda_
    opt = tfa.optimizers.AdamW(
                learning_rate=lr,
                    weight_decay=wd)
#    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    #Define callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                    factor=0.8,
                                                    patience=15)
    modelcheck = tf.keras.callbacks.ModelCheckpoint(
                                                    path_to_save + 'model',
                                                    monitor = 'val_loss',
                                                    verbose = 0,
                                                    save_best_only = True,
                                                    save_weights_only = True)
    if retrain == True:
        lambda_ = new_lambda
    def loss_MSE_mask(y_true, y_pred,mask=mask):
        y_true_masked = mask * y_true
        y_pred_masked = mask * y_pred
        loss = K.mean(((y_true_masked-y_pred_masked )**2))
        return loss

    def local_loss(y_true, y_pred,mask=mask):
        y_true_masked = mask * y_true
        y_pred_masked = mask * y_pred
        loss = K.mean(((y_true_masked-y_pred_masked )**2))
        return loss
    def global_loss(y_true, y_pred,mask=mask, lambda_ = lambda2):
        y_true_masked = mask * y_true
        y_pred_masked = mask * y_pred
        sum_y_true = K.mean(y_true_masked)
        sum_y_pred = K.mean(y_pred_masked)
        loss = (K.mean((sum_y_true - sum_y_pred)**2))
        return loss

    def PSD_loss(y_true, y_pred, mask =mask):
        y_true_masked = tf.reshape(y_true,(-1, 128, 128, N_output))
        y_pred_masked = tf.reshape(y_pred,(-1, 128, 128, N_output))

        
        PSD_y_true = create_power_spectra(y_true_masked, 128)
        PSD_y_pred = create_power_spectra(y_pred_masked, 128)
        
        return 1e-6*K.mean(K.abs(PSD_y_true - PSD_y_pred))
    def global_variance(y_true, y_pred,mask=mask, mu = mu):
        y_true_masked = mask * y_true
        y_pred_masked = mask * y_pred
        sum_y_true = K.std(y_true_masked)
        sum_y_pred = K.std(y_pred_masked)
        loss = (K.mean((sum_y_true - sum_y_pred)**2))
        return loss
    
    def tikhonov(y_true, y_pred,mask=mask, lambda_ = lambda2, nu = nu):
        y_true_masked = mask * y_true
        y_pred_masked = mask * y_pred
        sum_y_true = K.mean(y_true_masked)
        sum_y_pred = K.mean(y_pred_masked)
        return sum_y_pred**2

    def loss_variance(y_true, y_pred,mask=mask, lambda_ = lambda2, mu = mu, nu = nu):

        
        y_true_masked = mask * y_true
        y_pred_masked = mask * y_pred
        sum_y_true = K.mean(y_true_masked)
        sum_y_pred = K.mean(y_pred_masked)

        PSD_y_true = create_power_spectra(tf.reshape(y_true,(-1,128,128, N_output)), 128)
        PSD_y_pred = create_power_spectra(tf.reshape(y_true,(-1,128,128, N_output)), 128)
        
        loss = 1e-4*K.mean(K.abs(PSD_y_true - PSD_y_pred)) + K.mean(((y_true_masked-y_pred_masked )**2)) + lambda_*(K.mean((sum_y_true - sum_y_pred)**2)) 
        return loss

    def loss_sum (y_true, y_pred,mask=mask, lambda_ = lambda2):
        
        y_true_masked = mask * y_true
        y_pred_masked = mask * y_pred
        sum_y_true = K.mean(y_true_masked)
        sum_y_pred = K.mean(y_pred_masked)
        PSD_y_true = create_power_spectra(tf.reshape(y_true,(-1,128,128, N_output)), 128)
        PSD_y_pred = create_power_spectra(tf.reshape(y_true,(-1,128,128, N_output)), 128)
        loss = K.mean(((y_true_masked-y_pred_masked )**2)) + 0*K.mean(K.abs(PSD_y_true - PSD_y_pred))+lambda_*(K.mean((sum_y_true - sum_y_pred)**2))



        return loss
    def loss_MAE_mask(y_true, y_pred,mask=mask):
        y_true_masked = mask * y_true
        y_pred_masked = mask * y_pred
        loss = K.abs(y_true_masked-y_pred_masked )
        return loss
    if loss == 'sum':
        loss_mask = loss_sum
    if loss == 'variance':
        loss_mask = loss_variance
    if loss == 'MAE':
        loss_mask = loss_MAE_mask
    if loss == 'MSE':
        loss_mask = loss_MSE_mask
    #Compile the model
    model.compile(optimizer = opt, run_eagerly=True,  loss = loss_variance, metrics =[PSD_loss, local_loss, global_loss, global_variance, tikhonov])
    history = model.fit(train_dataset,
                        batch_size = batch_size,
                        validation_data = valid_dataset,
                        epochs = epochs,
                        callbacks = [early_stop, modelcheck],
                        verbose = verbose)
#                        callbacks = [callback,reduce_lr, log])

    print('### SAVE RESULTS ###')
    if retrain == False:
        np.save(path_to_save+'weights.npy', opt.get_weights())
        np.save(path_to_save+'global_loss.npy',history.history['global_loss'])
        np.save(path_to_save+'local_loss.npy',history.history['local_loss'])
        np.save(path_to_save+'loss.npy',history.history['loss'])
        np.save(path_to_save+'global_variance.npy',history.history['global_variance'])
        np.save(path_to_save+'tikhonov.npy',history.history['tikhonov'])
        np.save(path_to_save+'val_tikhonov.npy',history.history['val_tikhonov'])
        np.save(path_to_save+'val_global_variance.npy',history.history['val_global_variance'])
        np.save(path_to_save+'val_loss.npy',history.history['val_loss'])
        np.save(path_to_save+'val_global_loss.npy',history.history['val_global_loss'])
        np.save(path_to_save+'val_local_loss.npy',history.history['val_local_loss'])
        np.save(path_to_save+'PSD_loss.npy',history.history['PSD_loss'])
        np.save(path_to_save+'val_PSD_loss.npy',history.history['val_PSD_loss'])
