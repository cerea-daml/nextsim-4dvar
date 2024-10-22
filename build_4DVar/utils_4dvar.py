import numpy as np
import tensorflow as tf
from scipy.optimize import minimize, fmin_l_bfgs_b
from Models import UNet3
import matplotlib.pyplot as plt
import scipy as sc
from observations import observations

from build_dataset import dataset
from skimage.measure import block_reduce
#from background import load_background
import time


tf.config.run_functions_eagerly(True)
class FourDVar():
    '''
    Python class to test a surrogate model
    '''
    def __init__(self, model, mask, k,obs_op, N_cycle,starting_time, ftol,time_window, inflation, std_back, std_obs, timestep, save_grad, save_pred, path_to_save, path_to_data, N_x = 128, N_y = 128):
        tf.keras.backend.set_floatx('float64')
        
        self.H_obs = np.load('indices_Baffin.npy')
        self.H_obs = tf.cast(self.H_obs, tf.int32)

        #print(rmse(sit[:14*self.N_obs_per_window*self.N_cycle:14].numpy(), obs_field[:self.N_obs_per_window*self.N_cycle]))
        self.N_obs = np.shape(self.H_obs)[0]
        print(self.N_obs)
        self.R = np.load('R.npy')
        self.R = tf.cast(self.R, tf.float64)
        # Initialize constant and access to model
        self.model = model
        self.mask = mask
        self.k = k
        self.inflation = inflation
        self.N_cycle = N_cycle
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = self.N_x * self.N_y
        self.path_to_data = path_to_data
        self.path = path_to_save
        self.timestep = 1
        self.ftol = ftol
        self.save_pred = save_pred
        self.save_grad = save_grad
        #Normalization constant
        self.mean_input =0.38415005912623124
        self.std_input = 0.7710474897556033
        self.mean_output = -1.6270055141928728e-05
        self.std_output = 0.023030024601018818
    
        #Load mask 
        #self.mask = np.load('../../../data/transfer/mask.npy')
        #self.mask = block_reduce(mask, block_size = (4, 4, 1), func = np.min)
        #self.mask = 1 - self.mask
        self.mask_flatten = self.mask.flatten()
        print(np.shape(self.mask))
        #Get non-masked position
        self.unflatten = np.where((self.mask.flatten()>0))
        self.N_z_reduced = np.shape(self.unflatten[0])[0]
        self.mask_flatten = self.mask.flatten()
        mask_tf = tf.constant(self.unflatten[0], dtype=tf.int32)
        self.mask_expanded = tf.expand_dims(mask_tf, axis=-1)
        #Data assimilation constant
        self.window_length = time_window
        self.N_obs_per_window = 8
        self.starting_time = starting_time
        self.num_batches_per_window = self.window_length//self.N_obs_per_window
        self.std_obs = std_obs
        self.std_back = std_back
        self.obs_op = obs_op
        #a = tf.stack(a)

        #Load weights of the model
        loaded_s = self.model.load_weights(self.path + "model").expect_partial()  
        self.R = tf.reshape(self.R, (self.N_z_reduced, 1))

        #Tensorflow initialization
        self.state = tf.Variable(initial_value = np.zeros((1, self.N_z_reduced)), dtype = tf.float64 )
        self.x_0 = tf.Variable(initial_value = np.zeros((1, self.N_z_reduced)), dtype = tf.float64 )
        self.x_b = tf.Variable(initial_value = np.zeros((1, self.N_z_reduced)), dtype = tf.float64 )
        self.y = [tf.Variable(initial_value = np.zeros((1, self.N_z_reduced)), dtype = tf.float64) for k in range(self.N_obs_per_window*self.N_cycle + 1)]


    def normalize_input(self, x):
        return (x - self.mean_input) / self.std_input

    def reverse_normalize_output(self, x):
        return  x * self.std_output + self.mean_output 

    def reverse_normalize_input(self, x):
        return  x * self.std_input + self.mean_input 

    #Run the model during the complete assimilation window
    @tf.function
    def free_run_analysis(self, x, N_forcings, length):
        '''run the model during the assimilation window'''

        x = tf.reshape(x, (self.N_x, self.N_y)) 
        store_free_run = np.zeros((length, self.N_x, self.N_y))
        
        for l in range(0, length):

            store_free_run[l] = x


            x2 = tf.concat([tf.reshape(x, (self.N_x, self.N_y, 1)), self.forcings[N_forcings + l]], axis = 2)

            err = self.model(
                tf.reshape(x2 , (1, self.N_x, self.N_y, 4 * self.timestep + 6))
            )
            x = tf.reshape(err[0, :, :, 0], (self.N_x, self.N_y))
        return store_free_run

    #Run the model during the complete assimilation window
    @tf.function
    def free_run(self, x, N_forcings):
        '''run the model during the assimilation window'''    
        
        x = tf.identity(x)
        x_2 = tf.zeros((self.N_z), dtype = tf.float64)
        x_2 = tf.Variable(x_2)

        x_t = tf.reshape(tf.tensor_scatter_nd_update(x_2, self.mask_expanded, x[0]), (self.N_x, self.N_y))
 
        store_free_run = np.zeros((self.window_length, self.N_x, self.N_y))
        #x_t = self.mean + self.std * x_t
        for l in range(0, self.window_length):
            
            store_free_run[l] = x_t.numpy()


            x = tf.concat([tf.reshape(x_t, (self.N_x, self.N_y, 1)), self.forcings[N_forcings + l]], axis = 2)

            err = self.model(
                tf.reshape(x , (1, self.N_x, self.N_y, 4 * self.timestep + 6))
            )
            x_t = tf.reshape(err[0, :, :, 0], (self.N_x, self.N_y))
        return store_free_run

    #### FORWARD FUNCTION FOR 4DVAR MINIMIZATION ####
    @tf.function
    def forward(self, x_t, forcings):
        start_time_forward = time.time() 
        x = tf.identity(x_t)
        #x = self.mean + x*self.std
        x = tf.reshape(x, (1, self.N_z_reduced))

        x_2 = tf.zeros((self.N_z), dtype = tf.float64)
        x_2 = tf.Variable(x_2)
        
        x = tf.reshape(tf.tensor_scatter_nd_update(x_2, self.mask_expanded, x[0]), (self.N_x, self.N_y))

    
        #Run the model during assimilation window
        
        for l in range(self.num_batches_per_window):
            
            #Compute one iteration of the model
            x1 = tf.concat([tf.reshape(x, (self.N_x, self.N_y, 1)), self.forcings[l]], axis = 2)
            err = self.model(
                tf.reshape(x1 , (1, self.N_x, self.N_y, 4 * self.timestep + 6))
            )
            x = tf.reshape(err[0, :, :, 0], (self.N_x, self.N_y))
            
        x = tf.reshape(x, (self.N_z))
        x = tf.gather(x, self.unflatten[0])
        x = tf.reshape(x, (1, self.N_z_reduced))
        self.store_time_forward.append(time.time() - start_time_forward)
        return x
    #### OBSERVATION LOSS ####
    @tf.function
    def observation_loss(self, u, y):
        u_obs = tf.reshape(u, (self.N_z_reduced))
        y_obs = tf.reshape(y, (self.N_z_reduced))

        #d = tf.reshape(tf.gather(y_obs,self.H_obs) - tf.gather(u_obs,self.H_obs), (self.N_obs, 1))

        d = 1/self.R *tf.reshape(y_obs - u_obs, (self.N_z_reduced, 1))        
        dy_t = tf.transpose(d)@d
        return dy_t
    
    def observation_loss_single(self, u, y):
        d = y[0,6200] - u[0,6200]
        dy_t = 1/(self.std_obs)**2*d**2
        return dy_t

    def observation_loss_ten(self, u, y):
        d = y[0,6200:6210] - u[0,6200:6210]
        d = tf.reshape(d, (10, 1))
        dy_t = 1/(self.std_obs)**2*tf.transpose(d)@d
        return dy_t
    #### LOSS FUNCTION ####
    @tf.function
    def loss(self, x_0):
        
        x_t = tf.identity(x_0)
        #x_t = tf.reshape(x_t, (1, self.N_z_reduced))
        
        
        #Background loss
        dx  =  (x_t - self.x_b)/self.std_back
        J = 1/self.inflation_new*tf.math.reduce_sum(dx**2, axis = -1)
        self.back_J.append(J)
        
        obs_J = 0
    
        for k in range(1+ self.N*self.N_obs_per_window , (self.N + 1)*self.N_obs_per_window+1):
            x_t = self.forward(x_t, 
                    self.forcings[self.N*self.window_length + self.num_batches_per_window * (k- 1 - self.N*self.N_obs_per_window):
                                self.N*self.window_length + self.num_batches_per_window * (k-self.N*self.N_obs_per_window)])
            if self.obs_op == 'ten':
                obs_J += self.observation_loss_ten(x_t, tf.reshape(self.y[k], (1, self.N_z_reduced)))
            elif self.obs_op == 'single':
                obs_J += self.observation_loss_single(x_t, tf.reshape(self.y[k], (1, self.N_z_reduced)))
            else:
                obs_J += self.observation_loss(x_t, tf.reshape(self.y[k], (1, self.N_z_reduced)))
        #Save value of J
        
        self.obs_J.append(obs_J)
        J += obs_J
        self.J_min = J/2
        self.tot_J.append(J/2)
        return 1/2 * J

    ##### GRADIENT #####
    @tf.function
    def gradient(self, x_0):
        with tf.GradientTape() as tape:
            tape.reset()
            loss = self.loss(x_0)
            
        variables = [x_0]
        gradients = tape.gradient(loss, variables)
        self.gradients = gradients
        return gradients, loss

    def cost_grad(self, x_0):
        time_grad_0 = time.time()
        x = tf.Variable(initial_value = np.zeros((1, self.N_z_reduced)), dtype = tf.float64 )
        x.assign(tf.reshape(x_0, (1, self.N_z_reduced)))
        grad, loss  = self.gradient(x)
        self.store_time_grad.append(time.time()-time_grad_0)
        if self.save_grad == True:
            np.save(self.save_pred + 'grad/grad_{}_cycle{}.npy'.format(str(self.flag_grad), str(self.N)),grad[0].numpy())  
            self.flag_grad+=1
        
        return (loss.numpy(), grad[0].numpy())
        

    ##### RUN ANALYSIS #####(cycle_num, self.window_length, self.N_x, self.N_y)
    def run_analysis(self, cycle_num):
        self.save_J=np.zeros(cycle_num)
        self.j = 0
        self.flag_grad = 0
        #store_x_free = np.zeros((cycle_num, self.window_length, self.N_x, self.N_y))
        store_x_4Dvar = np.zeros((cycle_num, self.window_length, self.N_x, self.N_y))

        self.inflation_new = self.inflation
        for n in range(cycle_num):
            start_time = time.time()
            self.N = n
            print(n) 
            #Define bounds
            bounds = [(self.min, None) for i in range(8871)]
            
            #Perform minimisation
            print('start minimization')
            result =  minimize(self.cost_grad, self.state[0],bounds = bounds, method='L-BFGS-B', jac=True, options = {'disp':90,  'gtol': 1e-4, 'ftol':self.ftol})        
            self.result = result.x
            self.store_time_cycle[n] = time.time() - start_time
            np.save(self.save_pred + 'state_{}.npy'.format(str(n)), self.result)
            #Run free run and 4DVar run during one cycle
            #free = self.free_run(self.state, (self.N)*self.window_length)
            FourDRun = self.free_run(tf.reshape(result.x, (1, self.N_z_reduced)) , (self.N)*self.window_length)
            self.save_J[n]= self.J_min
            #Upload new initialization for the 4DVar
            #self.state = tf.reshape(tf.gather(
             #                   tf.reshape(free[-1], self.N_z), self.unflatten, axis = 0),(1, self.N_z_reduced))
            #Upload new x_b
            self.x_b=tf.reshape(tf.gather(
                                tf.reshape(FourDRun[-1], self.N_z), self.unflatten, axis = 0),(1, self.N_z_reduced))
            self.state = self.x_b
            np.save(self.save_pred + 'back_{}.npy'.format(str(n)), self.x_b)         
            #Store the trajectory
            self.inflation_new = self.inflation_new
            #store_x_free[n] = free
            store_x_4Dvar[n] = FourDRun
            self.flag_grad +=1     
        return store_x_4Dvar

    def test(self):
        
        '''Routine for applying a 4D-Var on neXtSIM SIT '''
        
        self.compt = 0
        self.tot_J = []
        self.obs_J = []
        self.back_J = []
        # Initialization
        PATH_TO_DATA = '../../../data/SIT_LR_full_state/'
        
        #Load dataset and observations object
        data = dataset(PATH_TO_DATA, self.N_obs_per_window, (self.window_length + 150)*self.N_cycle, starting_time = self.starting_time)
        obs = observations(PATH_TO_DATA, self.num_batches_per_window, (self.window_length + 100)*self.N_cycle, self.std_obs, starting_time = self.starting_time)
        
        print(np.shape(obs))
        #Get Observations, SIT and forcings
        obs_field, time_obs = obs.load_observation()
        sit, self.forcings, sit_input,time_data = data.load_data()

        #Store observations
        np.save(self.save_pred + 'sit_input.npy', np.array(sit_input))
        np.save(self.save_pred + 'obs.npy',np.array(obs_field))


        #Store minimal value for SIT
        self.min =  np.min(sit)
        
        #Put forcings in tf float 64
        self.forcings = tf.cast(self.forcings, tf.float64)      
        
        #Get initial state
        x_0 = tf.reshape(tf.cast(sit_input,dtype = tf.float64),(self.N_z))
        self.x_0.assign(tf.reshape(tf.gather(x_0, self.unflatten[0], axis = 0),(1, self.N_z_reduced)))
        
        self.state.assign(self.x_0.numpy())
        cycle_num = self.N_cycle


        
        self.x_b.assign(tf.reshape(tf.gather(x_0, self.unflatten[0], axis = 0),(1, self.N_z_reduced)))
        
        #Get observations
        
        for k in range(self.N_cycle * self.N_obs_per_window + 1):
            print(k)
            obs = tf.reshape(tf.cast(obs_field[k], dtype = tf.float64)*self.mask.squeeze(),(self.N_z))
            self.y[k].assign(tf.reshape(tf.gather(obs, self.unflatten[0], axis = 0),(1 , self.N_z_reduced)))
            

        start_time = time.time()
        self.store_time_cycle = np.zeros(cycle_num)
        self.store_time_grad = []
        self.store_time_forward=[]


        #Run analysis
        x_4D = self.run_analysis(cycle_num)
        np.save(self.save_pred + 'J.npy',np.array(self.tot_J))
        np.save(self.save_pred + 'J_obs.npy',np.array(self.obs_J))
        np.save(self.save_pred + 'J_back.npy',np.array(self.back_J))
        np.save(self.save_pred + 'J_min.npy',np.array(self.save_J))
        x_t_0 = tf.reshape(tf.cast(sit[0],dtype = tf.float64),(self.N_z))
        np.save(self.save_pred + 'time_cycle.npy',np.array(self.store_time_cycle))
        np.save(self.save_pred + 'time_grad.npy',np.array(self.store_time_grad))
        np.save(self.save_pred + 'time_forward.npy',np.array(self.store_time_forward))
#Launch long analysis forecast
        x_analysis_forecast = np.zeros((cycle_num-2, 90, self.N_x, self.N_y))
        x_analysis_forecast2 = np.zeros((cycle_num-4, 90, self.N_x, self.N_y))
        x_free_forecast = np.zeros((cycle_num-2, 90, self.N_x, self.N_y))
        for i in range(cycle_num-2):

            print(i)
            x_analysis_forecast[i] = self.free_run_analysis(x_4D[i, 0], i*self.window_length, 90)

            x_free_forecast[i] = self.free_run_analysis(tf.reshape(tf.cast(sit[i*self.window_length],dtype = tf.float64),(self.N_z)), i*self.window_length, 90)
        x_free = self.free_run_analysis(x_t_0, 0, self.window_length*self.N_cycle)
        return x_4D, x_free, sit, x_analysis_forecast, x_free_forecast

def rmse(x, y):
    return tf.math.sqrt(tf.reduce_mean((x - y)**2, axis = (1, 2)))


