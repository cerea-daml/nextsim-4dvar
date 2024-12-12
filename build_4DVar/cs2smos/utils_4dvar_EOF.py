import numpy as np
import tensorflow as tf
from scipy.optimize import minimize, fmin_l_bfgs_b
from Models import UNet3
import matplotlib.pyplot as plt
import scipy as sc
from real_observations import observations

from build_dataset import dataset
from skimage.measure import block_reduce



tf.config.run_functions_eagerly(True)
class FourDVar():
    '''
    Python class to run a 4D-Var minimization using EOFs
    '''
    def __init__(self, model, mask, k, N_cycle,starting_time,  ftol, obs_op, time_window, inflation, std_obs, timestep, save_grad, path_to_model, path_to_save, path_to_data, N_x = 128, N_y = 128):
        '''
        ----------------------------
        Parameters
        ----------------------------
        model: tf model, emulator model
        mask: land/sea-ice mask 
        k: int, EOF truncation index
        N_cycle: int, number of cycle in the assimilation
        starting_time: int, beginning of first assimilation window
        ftol: L-BFGS-B minimization criterion based on the cost function
        obs_op: str, observation operator parameters, 
                    - 'multi': observations on every point
                    - '<region>': observations on a given regions, example 'Barents'
                    - 'ten': observations on ten points
                    - 'single': observations on a single point
        time_window: int, length of the data assimilation window
        inflation: float, initial model inflation
        std_back: float, background covariance error
        std_obs: float, observation covariance error
        timestep: int, emulator input timestep number, default 1
        save_grad: bool, whether to save the gradient values
        save_pred: str, path to save results
        path_to_model: str, path to the emulator
        path_to_data: str, path to the observations and atmospheric forcings
        N_x, N_y, int, data shape
        '''''

        #By default minimization is done in tf.float 64
        tf.keras.backend.set_floatx('float64')
        
        #----------------
        # Access to model
        #----------------

        #Load model architecture and its weigth with the path
        self.model = model
        self.path_to_model = path_to_model
        self.timestep = 1
        loaded_s = self.model.load_weights(self.path_to_model + "model").expect_partial()


        #-----------------------
        # Mask and 1D-2D mapping
        #-----------------------

        #Load mask
        self.mask = mask
        #Flatten the mask to access 1D coordinates
        self.mask_flatten = self.mask.flatten()
        #Get non-masked position
        self.unflatten = np.where((self.mask.flatten()>0))
        #Number of non-masked position
        self.N_z_reduced = np.shape(self.unflatten[0])[0]
        #Mask conversion to tensorflow for iteration of the surrogate
        mask_tf = tf.constant(self.unflatten[0], dtype=tf.int32)
        #Add one channel to the mask for the surrogate
        self.mask_expanded = tf.expand_dims(mask_tf, axis=-1)        


        #----------------------------------------
        # Constants and parameters initialization
        #----------------------------------------
        #Truncation index
        self.k = k
        #x_bar for projection onto the EOFs
        self.x_bar = np.load('../4dvar/EOF/mean_norm.npy')
        #Load the EOF with truncation index
        self.EOF = np.load("../4dvar/EOF/EOFs_y8_norm.npy")[:,:self.k]

        #Model inflation
        self.inflation = inflation

        #Number of cycles
        self.N_cycle = N_cycle

        #Shape of 2D fields
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = self.N_x * self.N_y

        #Path to data
        self.path_to_data = path_to_data
    
        #L-BFGS stopping criterion
        self.ftol = ftol

        #Starting time of 1st DAW
        self.starting_time = starting_time

        #Path to save results
        self.save_pred = path_to_save

        #Bool to save gradient of L-BFGS
        self.save_grad = save_grad

        #Normalization constant
        self.mean_input =0.38415005912623124
        self.std_input = 0.7710474897556033
        self.mean_output = -1.6270055141928728e-05
        self.std_output = 0.023030024601018818
        
        #--------------------------
        #Data assimilation constant
        #--------------------------

        #Type of observation
        self.obs_op = obs_op
        
        #DA window length
        self.window_length = time_window

        #Number of obs per window
        self.N_obs_per_window = 1

        #Number of iterations of the surrogate between 2 observations
        self.num_batches_per_window = self.window_length//self.N_obs_per_window
        #Observation covariance error
        self.std_obs = std_obs

        #Load matrix of observation covariance error and cast in tf float64
        self.R = np.load('R.npy')
        self.R = tf.cast(self.R, tf.float64)

        #Define observation operator and observation covariance error
        if self.obs_op not in ['multi', 'single', 'ten']:
            self.H_obs = np.load('indices_'+self.obs_op+'.npy')
            self.H_obs = tf.cast(self.H_obs, tf.int32)
            self.N_obs = np.shape(self.H_obs)[0]
            self.R = tf.gather(self.R,self.H_obs)
            self.R = tf.reshape(self.R, (self.N_obs, 1))
        else:
            self.R = tf.reshape(self.R, (self.N_z_reduced, 1))

        #-------------------------
        #Tensorflow initialization
        #-------------------------

        #Variable for L-BFGS minimization
        self.state = tf.Variable(initial_value = np.zeros((1, self.k)), 
                                dtype = tf.float64 )
        #State first guess
        self.x_0 = tf.Variable(initial_value = 
                                np.zeros((1, self.N_z_reduced)), 
                                dtype = tf.float64 )
        #State background
        self.x_b = tf.Variable(initial_value = 
                                np.zeros((1, self.N_z_reduced)), 
                                dtype = tf.float64 )
        #Observations
        self.y = [tf.Variable(initial_value = 
                                np.zeros((1, self.N_z_reduced)), 
                                dtype = tf.float64) 
                                for k in range(
                                    self.N_obs_per_window*self.N_cycle + 1)]


    # Normalization functions
    def normalize_input(self, x):
        return (x - self.mean_input) / self.std_input

    def reverse_normalize_output(self, x):
        return  x * self.std_output + self.mean_output 

    def reverse_normalize_input(self, x):
        return  x * self.std_input + self.mean_input 


    #Run the model during the complete assimilation window
    @tf.function
    def free_run_analysis(self, x, N_forcings, length):
        '''run the model during the assimilation window
        -------------------------
        Parameters
        -------------------------
        x: array (N_x, N_y), initial SIT field
        N_forcings: int, index to access corresponding 
                        forcings for the first iteration 
                        of the emulator
        length: int,  number of iterations of the emulator

        ------------------------
        Output
        ------------------------
        array with <length> iterations of the emulator with initial state <x>
        '''

        #Reshape SIT field to correct shape
        x = tf.reshape(x, (self.N_x, self.N_y)) 

        #Initialize array for results save
        store_free_run = np.zeros((length, self.N_x, self.N_y))
        
        #Iteration of the  emulator for length
        for l in range(0, length):
            
            #Store state
            store_free_run[l] = x

            #Concatenate SIT field with its atmospheric forcings
            x2 = tf.concat([tf.reshape(x, (self.N_x, self.N_y, 1)), 
                                self.forcings[N_forcings + l]], 
                                axis = 2)

            #Application of the emulator with correct shape for the input
            err = self.model(
                            tf.reshape(x2 , 
                            (1, self.N_x, self.N_y, 4 * self.timestep + 6))
                            )
            #Get first output and reshape it
            x = tf.reshape(err[0, :, :, 0], (self.N_x, self.N_y))
        
        return store_free_run
    
    #Run the model with ECMWF atmospheric forecast
    @tf.function
    def run_forecast(self, x, forcings, length):
        '''run the model with ECMWF forecast
         -------------------------
        Parameters
        -------------------------
        x: array (N_x, N_y), initial SIT field
        forcings: array, ECWMF forecast for the correct time and length
        length: int,  number of iterations of the emulator

        ------------------------
        Output
        ------------------------
        array with <length> iterations of the emulator with initial state <x>       '''

        #Reshape initial state
        x = tf.reshape(x, (self.N_x, self.N_y))

        #Initialize array for results save
        store_free_run = np.zeros((length, self.N_x, self.N_y))

        #Iteration of the  emulator for length
        for l in range(0, length):

            #Store state
            store_free_run[l] = x

            #Concatenate SIT field with its atmospheric forcings
            x2 = tf.concat([tf.reshape(x, (self.N_x, self.N_y, 1)), forcings[l]], axis = 2)
            
            #Application of the emulator with correct shape for the input
            err = self.model(
                            tf.reshape(x2 , 
                            (1, self.N_x, self.N_y, 4 * self.timestep + 6))
                            )

            #Get first output and reshape it
            x = tf.reshape(err[0, :, :, 0], (self.N_x, self.N_y))

        return store_free_run


    #Run the model during the complete assimilation window
    @tf.function
    def free_run(self, x, N_forcings):
        
        '''run the model during the assimilation window (self.window_length iterations)
        ------------------------
        Parameters
        ------------------------
        x: array of shape (1, self.N_z_reduced) for initial state of the forecast
        N_forcings: int, index to access corresponding
                        forcings for the first iteration
                        of the emulator
        -----------------------
        Output
        -----------------------
        array with <self.window_length> iterations of the emulator with initial state <x>
        '''    

        #-------------------------------
        #Map from 1D array to 2D masked array
        #------------------------------

        #Put x as tensor
        x = tf.identity(x)

        #Initialize 2D vector and create a Variable type
        x_2 = tf.zeros((self.N_z), dtype = tf.float64)
        x_2 = tf.Variable(x_2)

        #Map from 1D to 2D with self.mask_expanded mask 
        x_t = tf.reshape(tf.tensor_scatter_nd_update(x_2, self.mask_expanded, x[0]), (self.N_x, self.N_y))
 
        #Initialize results store
        store_free_run = np.zeros((self.window_length, self.N_x, self.N_y))

        #Iteration of the  emulator for self.window_length
        for l in range(0, self.window_length):
            
            #Store state
            store_free_run[l] = x_t.numpy()

            #Concatenate SIT field with its atmospheric forcings
            x = tf.concat([tf.reshape(x_t, (self.N_x, self.N_y, 1)), self.forcings[N_forcings + l]], axis = 2)

            #Application of the emulator with correct shape for the input
            err = self.model(
                            tf.reshape(x , 
                            (1, self.N_x, self.N_y, 4 * self.timestep + 6))
                            )
            #Get first output and reshape it
            x_t = tf.reshape(err[0, :, :, 0], (self.N_x, self.N_y))

        return store_free_run

    #------------------------------------------------
    #### FORWARD FUNCTION FOR 4DVAR MINIMIZATION ####
    #------------------------------------------------
    
    @tf.function
    def forward(self, x_t, forcings):
        '''run the model between 2 observations (self.num_batches_per_window)
        ------------------------
        Parameters
        ------------------------
        x: array of shape (1, self.N_z_reduced) for initial state of the forecast
        N_forcings: int, index to access corresponding
                        forcings for the first iteration
                        of the emulator
        -----------------------
        Output
        -----------------------
        array with <self.window_length> iterations of the emulator with initial state <x>
        '''

        #Put x as tensor and reshape it
        x = tf.identity(x_t)
        x = tf.reshape(x, (1, self.N_z_reduced))

        #Initialize 2D vector and create a Variable type
        x_2 = tf.zeros((self.N_z), dtype = tf.float64)
        x_2 = tf.Variable(x_2)
        
        #Map from 1D to 2D with self.mask_expanded mask
        x = tf.reshape(tf.tensor_scatter_nd_update(x_2, self.mask_expanded, x[0]), (self.N_x, self.N_y))

        #Run the model between two observations        
        for l in range(self.num_batches_per_window):
            
            #Concatenate SIT field with its atmospheric forcings
            x1 = tf.concat([tf.reshape(x, (self.N_x, self.N_y, 1)), self.forcings[l]], axis = 2)
            
            #Application of the emulator with correct shape for the input
            err = self.model(
                            tf.reshape(x1 , 
                            (1, self.N_x, self.N_y, 4 * self.timestep + 6))
                            )
            
            #Get first output and reshape it
            x = tf.reshape(err[0, :, :, 0], (self.N_x, self.N_y))
        
        #Return final state as 1D unmasked array of shape (1, self.N_z_reduced)
        x = tf.reshape(x, (self.N_z))
        x = tf.gather(x, self.unflatten[0])
        x = tf.reshape(x, (1,self.N_z_reduced))
        
        return x

    ##########################
    #### OBSERVATION LOSSES ####
    ##########################
    @tf.function
    def observation_loss(self, u, y):
        ''' Compute observation loss between u and y
        '''

        # In case of single observation
        if self.obs_op=='single':
            #Pick single obs and compute the diff between u and y
            d = y[0,6200] - u[0,6200]

            #Compute the associated error
            dy_t = 1/(self.std_obs)**2*d**2
            return dy_t

        #In case of ten observations
        elif self.obs_op=='ten':
            #Pick ten obs and compute the diff between u and y
            d = y[0,6200:6210] - u[0,6200:6210]

            #Reshape it
            d = tf.reshape(d, (10, 1))

            #Compute the associated error
            d = tf.reshape(d, (10, 1))
            dy_t = 1/(self.std_obs)**2*tf.transpose(d)@d
            return dy_t

        else:
            #Reshape u and y
            u_obs = tf.reshape(u, (self.N_z_reduced))
            y_obs = tf.reshape(y, (self.N_z_reduced))

            #If observations on every grid-cells
            if self.obs_op=='multi':
                #Compute the difference
                d = 1/self.R * tf.reshape(y_obs - u_obs, (self.N_z_reduced , 1))
                #Compute the associated error
                dy_t = tf.transpose(d)@d

            #if observation on a given region
            else:
                #Gather obs on region grid-cells and compute the difference
                d = 1/self.R*tf.reshape(tf.gather(y_obs,self.H_obs) - tf.gather(u_obs,self.H_obs), (self.N_obs, 1))
                #Compute the associated error
                dy_t = tf.transpose(d)@d
            return dy_t

    #######################
    #### LOSS FUNCTION ####
    #######################

    
    @tf.function
    def loss(self, u_0):
        '''
        Compute the cost function for the 4D-Var minimization
        --------------
        Parameters: 
        --------------
        u_0, tf Variable, initial state of the system for the minimization

        --------------
        Output
        --------------
        J: tf float, cost function on a given DAW
        '''

        #Recopy the initial state and reshape it
        u = tf.identity(u_0)
        u = tf.reshape(u,  (self.k, 1))
        
        
        #Compute background loss term
        back_J  = tf.transpose(u)@u
        #Store background loss term
        self.back_J.append(1/self.inflation_new*back_J)
        
        #Initialize observation loss term
        obs_J = 0 

        #Project u from EOF basis to real space
        x = tf.reshape(self.x_bar, (1, self.N_z_reduced)) + tf.reshape(self.EOF@u, (1, self.N_z_reduced))
        
        #Apply iteratively the emulator and computer observation loss
        for k in range(1+ self.N*self.N_obs_per_window , (self.N + 1)*self.N_obs_per_window+1):
            
            #Application of the emulator with the correct associated forcings
            x_t = self.forward(x, 
                                self.forcings[self.N*self.window_length + 
                                            self.num_batches_per_window * (k- 1 - self.N*self.N_obs_per_window):
                                            self.N*self.window_length + 
                                            self.num_batches_per_window * (k-self.N*self.N_obs_per_window)])
            #Add k-th observation error term    
            obs_J += self.observation_loss(x_t, tf.reshape(self.y[k], (1, self.N_z_reduced)))

        #Save value of observation loss term        
        self.obs_J.append(obs_J)
        
        #Compute total cost function
        J = 1/self.inflation_new*back_J + obs_J
        
        #Save total cost function
        self.tot_J.append(J/2)
        
        #Divide all cost function by two
        self.J = J/2
        self.Jo = obs_J/2
        self.Jb = 1/self.inflation_new*back_J/2
        
        #Return cost function
        return 1/2 * J

    ####################
    ##### GRADIENT #####
    ####################


    @tf.function
    def gradient(self, x_0):
        '''
        function to retrieve the cost function and its associated gradient
        ---------------
        Inputs
        ---------------
        x_0, tf variable of size (1, self.k) which is the variable to minimize

        --------------
        Outputs
        -------------
        gradients : tensorflow gradient of the cost function
        loss: cost function associated to x_0
        '''

        with tf.GradientTape() as tape:
            #Reinitialize the gradient
            tape.reset()
            #Compute the loss
            loss = self.loss(x_0)
        
        #Define the variable to compute the gradient
        variables = [x_0]

        #Compute the gradient with regard to <variables>
        gradients = tape.gradient(loss, variables)

        #Retain gradients
        self.gradients = gradients
        return gradients, loss

    def cost_grad(self, x_0):

        '''
        Wrapper for gradient computation
        ---------------
        Inputs
        ---------------
        x_0, tf variable of size (1, self.k) which is the variable to minimize

        --------------
        Outputs
        -------------
        gradients : numpy gradient of the cost function
        loss: numpy float, cost function associated to x_0
        '''

        #Define x_0 as a tf Variable
        x = tf.Variable(initial_value = np.zeros((1, self.k)), dtype = tf.float64 )
        x.assign(tf.reshape(x_0, (1, self.k)))

        #Get gradient and loss as tensorflow values
        grad, loss  = self.gradient(x)

        #Saving the gradients
        if self.save_grad == True:
            np.save(self.save_pred + 'grad/grad_{}_cycle{}.npy'.format(str(self.flag_grad), str(self.N)),grad[0].numpy())  
        
        return (loss.numpy(), grad[0].numpy())
        
    ############################
    ##### RUN ASSIMILATION #####
    ############################
    def run_analysis(self, cycle_num):

        '''
        Function to run the 4D-Var assimilation
        ----------------
        Inputs
        ----------------
        cycle_num: int, number of assimilation cycle

        ----------------
        Outputs
        store_x_4Dvar: array, analysis onto the full assimilation
        ---------------

        '''

        #Initialize cost function save array
        self.J_save = np.zeros(cycle_num)

        #Initialize flags for grad and cost function save
        self.j = 0
        self.flag_grad = 0

        #Initialize analysis array
        store_x_4Dvar = np.zeros((cycle_num, self.window_length, self.N_x, self.N_y))
        
        #Define inflation factor
        self.inflation_new = self.inflation
        

        #Cycle over assimilation cycles
        for n in range(cycle_num):

            #Current cycle number
            self.N = n  
            print('Cycle {}'.format(str(n)))

            #Perform minimisation
            print('Start minimization')
            result =  minimize(self.cost_grad, self.state[0], 
                                method='L-BFGS-B', jac=True, 
                                options = {'disp':90,  'gtol': 1e-4, 'ftol':self.ftol})  

            #Result and save of the minimization
            self.result = result.x
            np.save(self.save_pred + 'state_{}.npy'.format(str(n)), self.result)

            #Project analysis result onto real space
            self.x_0 = tf.reshape(self.x_bar, (self.N_z_reduced)) + tf.reshape(self.EOF@tf.reshape(self.result, (self.k, 1)), (self.N_z_reduced))
            
            #Run 4DVar run during one cycle
            FourDRun = self.free_run(tf.reshape(self.x_0, (1, self.N_z_reduced)) , (self.N)*self.window_length)
            
            #Save cost function at the end of the cycle
            self.J_save[n]= self.J

            #Upload new initialization for the 4DVar
            #Reshape last analysis term in 1D-unmasked array
            self.x_b=tf.reshape(tf.gather(
                                tf.reshape(FourDRun[-1], self.N_z), 
                                self.unflatten, axis = 0),(1, self.N_z_reduced))
            
            #Project onto the EOFs
            self.state.assign(tf.reshape(self.EOF.T@tf.reshape(self.x_b - self.x_bar, (self.N_z_reduced, 1)), (1, self.k)))
            
            #Save first guess
            np.save(self.save_pred + 'back_{}.npy'.format(str(n)), self.x_b)         
            
            #Store the trajectory
            store_x_4Dvar[n] = FourDRun

            #Increase flag for grad save
            self.flag_grad+=1
            
        return store_x_4Dvar

    def test(self):
        
        '''Wrapper for applying a 4D-Var on neXtSIM SIT 
        -------------------
        Outputs
        ------------------
        x_4D: analysis of the 4D-Var 
        x_free, free run of the emulator
        obs_field, observations 
        x_analysis_forecast, analysis + forecast at each cycle
        x_free_forecast,  forecast starting from the observations
        '''
        
        #Initialize array for cost function saving 
        self.tot_J = []
        self.obs_J = []
        self.back_J = []
    
        
        #Load dataset and observations object
        data = dataset(self.path_to_data, starting_time = self.starting_time)
        obs = observations(self.std_obs, starting_time = self.starting_time, frequency = self.num_batches_per_window)
        
        #Get Observations, SIT and forcings
        obs_field = obs.load_observation()
        sit, self.forcings, sit_input,time_data, forcings_forecast = data.load_data()

        #Save observations 
        np.save(self.save_pred + 'obs.npy',obs_field)

        #Store minimal value for SIT
        self.min =  np.min(sit)
        
        #Put forcings in tf float 64
        self.forcings = tf.cast(self.forcings, tf.float64)      
        
        #Get initial state
        x_0 = tf.reshape(tf.cast(obs_field[0],dtype = tf.float64),(self.N_z))
        self.x_0.assign(tf.reshape(tf.gather(x_0 + np.random.normal(0, 0.0, (self.N_z)), self.unflatten[0], axis = 0),(1, self.N_z_reduced)))
        
        #Move to perturbation space
        self.w_0 = self.EOF.T@tf.reshape(self.x_0 - tf.reshape(self.x_bar, (1, self.N_z_reduced)), (self.N_z_reduced, 1))
        
        #Define cycle number
        cycle_num = self.N_cycle

        #Initialize first guess
        self.x_b.assign(tf.reshape(self.x_bar, (1, self.N_z_reduced)))
        self.w_b = tf.reshape(self.EOF.T@tf.reshape(self.x_b, (self.N_z_reduced, 1)),(1, self.N_z_reduced))
        
        #Initialize function to minimize 
        self.state.assign(self.w_b.numpy())
        
        #Get observations and cast them in tensorflow
        for k in range(self.N_cycle * self.N_obs_per_window + 1):
            obs = tf.reshape(tf.cast(obs_field[k], dtype = tf.float64)*self.mask.squeeze(),(self.N_z))
            self.y[k].assign(tf.reshape(tf.gather(obs, self.unflatten[0], axis = 0),(1 , self.N_z_reduced)))
            

        #Run analysis
        x_4D = self.run_analysis(cycle_num)


        #Save cost function
        np.save(self.save_pred + 'J.npy',np.array(self.tot_J))
        np.save(self.save_pred + 'J_obs.npy',np.array(self.obs_J))
        np.save(self.save_pred + 'J_back.npy',np.array(self.back_J))
        np.save(self.save_pred + 'J_min.npy',np.array(self.J_save))
        
        #Launch long analysis forecast
        x_analysis_forecast = np.zeros((cycle_num - 1, 35, self.N_x, self.N_y))
        x_free_forecast = np.zeros((cycle_num-1, 35, self.N_x, self.N_y))

        #Perform forecast on the analysis and on the observations
        for i in range(cycle_num-1):

            #Perform forecast in DAW
            x_analysis_forecast[i, :16] = self.free_run_analysis(x_4D[i, 0], i*self.window_length, 16)

            #Cast ECMWF forcings in tf
            forcing_forecast = tf.cast(forcings_forecast[i], dtype = tf.float64)

            #Perform forecast outside DAW with ECMWF forcings
            x_analysis_forecast[i,16:] = self.run_forecast(x_analysis_forecast[i, 15], forcing_forecast, 19)

            #Perform forecast starting from obs
            #x_free_forecast[i, :16] = self.free_run_analysis(tf.reshape(tf.cast(obs_field[i*8],dtype = tf.float64),(self.N_z)), i*self.window_length, 16)
            #Continue with ECMWF forecast
            #x_free_forecast[i,16:] = self.run_forecast(x_free_forecast[i, 15], forcing_forecast, 19)
        
        #Free run across all DAW 
        x_free = self.free_run_analysis(sit_input, 0, self.window_length*self.N_cycle)

        return x_4D, x_free, obs_field, x_analysis_forecast, x_free_forecast


