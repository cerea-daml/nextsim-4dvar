import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b
from skimage.measure import block_reduce

import numpy.ma as ma
import matplotlib.pyplot as plt
import cmocean 
import xarray as xr
from tqdm import trange
from skimage.measure import block_reduce
from scipy.spatial import cKDTree
#plt.style.use('../../../../../cerea_raid/users/durandc/presentation.mplstyle')

class create_dataset_from_nextsim_outputs : 
    def __init__(self, N_res):
        self.N_res = N_res

    def prepare_data(self, sit) :
        N = np.shape(sit)[0]
        x_t = []
        for i in trange(N):
            x_t.append( ma.getdata(sit[i]).reshape((603,528,1)))

        N_shape = int(512/self.N_res)
        x_t = np.array(x_t)

        x_t = x_t[:,91:, 8:-8]

        mask = ma.getmaskarray(sit[0])
        mask = mask.reshape((603,528,1))
        mask = mask[91:, 8:-8]
        mask2 = block_reduce(mask, block_size=(self.N_res, self.N_res, 1), func = np.min)

        x_t = np.where(mask==True, 0, np.array(x_t))

        if self.N_res > 1 :
            x_coarse = np.zeros((np.shape(x_t)[0], N_shape, N_shape, 1))

            for i in trange(np.shape(x_t)[0]):
                x_coarse[i] =  block_reduce(x_t[i], block_size=(self.N_res, self.N_res, 1), func=np.mean)

            x_t = x_coarse
        return x_t, mask2
        
    def make_sit(self):
        

        N_shape = int(512/self.N_res)


        liste = []
        years_train = ['2009','2010','2011','2012','2013','2014','2015','2016']
        months = ['01','02','03','04','05','06','07','08','09','10','11','12']
        for y in years_train:
            for m in months : 
                liste.append('../Data/Moorings_'+y+'m'+m+'.nc')
        print('Build Train')
        ds_train = xr.open_mfdataset(liste)
        sit_train = ds_train.sit
        time_train = ds_train.time
        sit_train = sit_train.to_masked_array()
        print('Build Val')
        ds_val = xr.open_mfdataset('../Data/Moorings_2017m*.nc')
        sit_val = ds_val.sit
        time_val = ds_val.time
        sit_val = sit_val.to_masked_array()
        print('Build Test')
        liste2=[]
        years_train = ['2006','2007','2008']
        months = ['01','02','03','04','05','06','07','08','09','10','11','12']
        for y in years_train:
            for m in months :
                liste2.append('../Data/Moorings_'+y+'m'+m+'.nc')
        ds_test = xr.open_mfdataset(liste2)
        sit_test = ds_test.sit
        time_test  = ds_test.time
        sit_test = sit_test.to_masked_array()
        np.save('time_test_2006_2009.npy', time_test)        
        print('prepare test')
        x_test, mask = self.prepare_data(sit_test)
        print(np.shape(mask))
        print('prepare val')
        x_val,_  = self.prepare_data(sit_val)
        print('prepare train')
        x_train,_ = self.prepare_data(sit_train)
    
        np.save('mask.npy',mask)
        x_tr = x_train

        y_tr =np.concatenate([x_train[8:-2]-x_train[6:-4] , x_train[10: ] - x_train[6:-4]], axis = 3)
        N_tr = np.shape(x_tr)[0]
        x_train = np.zeros((N_tr-10,N_shape,N_shape,2))
        y_train = y_tr
        for i in range(N_tr-10):
            x_train[i,:,:,0] = x_tr[i+4].squeeze()
            x_train[i,:,:,1] = x_tr[i+6].squeeze()
        y_v = np.concatenate([x_val[8:-2]-x_val[6:-4] , x_val[10:] - x_val[6:-4]], axis = 3)
        x_v = x_val


        N_val = np.shape(x_v)[0]
        x_val = np.zeros((N_val-10,N_shape,N_shape,2))
        y_val = y_v
        for i in range(N_val-10): 
            x_val[i,:,:,0] = x_v[i+4].squeeze()
            x_val[i,:,:,1] = x_v[i+6].squeeze()

        y_te =np.concatenate([x_test[8:-2]-x_test[6:-4] , x_test[10:] - x_test[6:-4]], axis = 3)
        x_te = x_test   

        N_test = np.shape(x_te)[0]
        x_test = np.zeros((N_test-10,N_shape,N_shape,2))
        y_test = y_te
        for i in range(N_test-10):
            x_test[i,:,:,0] = x_te[i+4].squeeze()
            x_test[i,:,:,1] = x_te[i+6].squeeze()


        source = xr.open_dataset('../Data/Moorings_2009m01.nc')
        lat_source = source.variables['latitude'][91:, 8:-8]
        lon_source = source.variables['longitude'][91:, 8:-8]
        lat_source = lat_source[::self.N_res,::self.N_res]
        lon_source = lon_source[::self.N_res,::self.N_res]
        
        
        mask = np.multiply(mask, 1.)
        self.mask = mask
        print('VAL INPUTS')
        val1 = xr.Dataset(coords={"time": time_val[10:],"prec":[3,4], "lat": (["x","y"], lat_source),"lon": (["x","y"], lon_source)})
        val1["inputs_sit"] = (['time','lat', 'lon','prec'],  x_val.squeeze())
        val1.to_netcdf('CoarseResolution/val_inputs.nc', mode='w')
        print('VAL OUTPUTS')
        val2 = xr.Dataset(coords={"time": time_val[10:], "prec":[1,2], "lat": (["x","y"], lat_source),"lon": (["x","y"], lon_source)})
        val2["outputs_sit"] = (['time','lat', 'lon','prec'],  y_val.squeeze())
        val2.to_netcdf('CoarseResolution/val_outputs.nc', mode='w')
        print('TEST_INPUTS')
        test1 = xr.Dataset(coords={"time": time_test[10:],"prec":[3,4], "lat": (["x","y"], lat_source),"lon": (["x","y"], lon_source)})
        test1["inputs_sit"] = (['time','lat', 'lon','prec'],  x_test.squeeze())
        test1.to_netcdf('CoarseResolution/test_inputs.nc', mode='w')
        print('TEST OUTPUTS')
        test2 = xr.Dataset(coords={"time": time_test[10:], "prec":[1,2],"lat": (["x","y"], lat_source),"lon": (["x","y"], lon_source)})
        test2["outputs_sit"] = (['time','lat', 'lon', 'prec'],  y_test.squeeze())
        test2.to_netcdf('CoarseResolution/test_outputs.nc', mode='w')
        print(test2) 

        print("Train")
        
        train1 = xr.Dataset(coords={"time": time_train[10:],"prec":[3,4], "lat": (["x","y"], lat_source),"lon": (["x","y"], lon_source)})
        train1["inputs_sit"] = (['time','lat', 'lon','prec'],  x_train.squeeze())
        train1.to_netcdf('CoarseResolution/train_inputs.nc', mode='w')
        print('VAL OUTPUTS')
        print(train1)
        train2 = xr.Dataset(coords={"time": time_train[10:], "prec":[1,2],"lat": (["x","y"], lat_source),"lon": (["x","y"], lon_source)})
        train2["outputs_sit"] = (['time','lat', 'lon','prec'],   y_train.squeeze())
        train2.to_netcdf('CoarseResolution/train_outputs.nc', mode='w')
        return mask
        

    def interpolate(self, time, inputs, dataset, data_name, target_shape, d, inds):
        nt = time.shape[0]
        tmp = {}

        tmp[data_name] = []

        for t in trange(0, nt):

            var = dataset[data_name][t].values.flatten()[inds]

            var.shape = target_shape.shape
            var = block_reduce(var[91:, 8:-8], block_size=(self.N_res, self.N_res), func=np.mean)
            tmp[data_name].append(var)

        return tmp

    def lon_lat_to_cartesian(self, lon, lat):
        # WGS 84 reference coordinate system parameters
        A = 6378.137  # major axis [km]
        E2 = 6.69437999014e-3  # eccentricity squared

        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        # convert to cartesian coordinates
        r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
        x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
        y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
        z = r_n * (1 - E2) * np.sin(lat_rad)
        return x, y, z

    def create_forcings_dataset(self, nextsim_data, variable, data_name):
        source = xr.open_dataset(nextsim_data)
        lat_source = source.variables["latitude"]
        lon_source = source.variables["longitude"]
        print("OPEN FORCINGS DATASET")

        liste = []
        years_train = ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]
        for y in years_train:
            liste.append("../Forcings/ERA5_" + variable + "_y" + y + ".nc")
        dataset_train = xr.open_mfdataset(liste)
        dataset_train.fillna(0)

        time_train = dataset_train.time.data

        dataset_val = xr.open_dataset("../Forcings/ERA5_" + variable + "_y2017.nc")
        dataset_val.fillna(0)


        liste2 = []
        years_test = ['2020','2021']
        for y in years_test:
            liste2.append("../Forcings/ERA5_" + variable + "_y" + y + ".nc")
        dataset_test = xr.open_mfdataset(liste2)
        dataset_test.fillna(0)

        time_test = dataset_test.time.data
        time_val = dataset_val.time.data

    
        lat_target = dataset_val.latitude
        lon_target = dataset_val.longitude

        lon_target2d, lat_target2d = np.meshgrid(lon_target, lat_target)

        print("CONVERT LAT, LON TO CARTESIAN GRID")
        xt, yt, zt = self.lon_lat_to_cartesian(
            lon_source.values.flatten(), lat_source.values.flatten()
               )
        xs, ys, zs = self.lon_lat_to_cartesian(lon_target2d.flatten(), lat_target2d.flatten())

        print("INTERPOLATE")
        tree = cKDTree(np.column_stack((xs, ys, zs)))
        d, inds = tree.query(np.column_stack((xt, yt, zt)), k=1)

        forcings_train = self.interpolate(
            time_train, source, dataset_train, data_name, lat_source, d, inds
        )
        forcings_val = self.interpolate(
            time_val, source, dataset_val, data_name, lat_source, d, inds
        )
        forcings_test = self.interpolate(
            time_test, source, dataset_test, data_name, lat_source, d, inds
        )

        print("CREATE DATASETS")
        lat_source = lat_source[91:, 8:-8]
        lon_source = lon_source[91:, 8:-8]

        lat_source = lat_source[::self.N_res, ::self.N_res]
        lon_source = lon_source[::self.N_res, ::self.N_res]

        print(np.shape(lat_source))
        print(np.shape(lon_source))
        dataset_train = xr.Dataset(
            coords={
            "time": time_train[10:],
            "prec": [3, 4, 5, 6],
            "lat": (["x", "y"], lat_source),
            "lon": (["x", "y"], lon_source),
            }
        )
        dataset_val = xr.Dataset(
            coords={
            "time": time_val[10:],
            "prec": [3, 4, 5, 6],
            "lat": (["x", "y"], lat_source),
            "lon": (["x", "y"], lon_source),
            }
        )

        dataset_test = xr.Dataset(
            coords={
                "time": time_test[591*2+6:-4:2],
            "prec": [3, 4, 5, 6],
            "lat": (["x", "y"], lat_source),
            "lon": (["x", "y"], lon_source),
            }
        )
        print("Add variables")

        dataset_train[data_name] = (
            ["time", "prec", "lat", "lon"],
            np.array(
                [
                np.array(forcings_train[data_name])[4:-6],
                np.array(forcings_train[data_name])[6:-4],
                np.array(forcings_train[data_name])[7:-3],
                np.array(forcings_train[data_name])[8:-2],
                ]
            ).transpose((1, 0, 2, 3)),
        )

        dataset_val[data_name] = (
            ["time", "prec", "lat", "lon"],
            np.array(
                [
                np.array(forcings_val[data_name])[4:-6],
                np.array(forcings_val[data_name])[6:-4],
                np.array(forcings_val[data_name])[7:-3],
                np.array(forcings_val[data_name])[8:-2],
                ]
            ).transpose((1, 0, 2, 3)),
        )

        print(np.shape(forcings_test[data_name]))
        print(time_test[591*2+6])
        dataset_test[data_name] = (
            ["time", "prec", "lat", "lon"],
            np.array(
                [
                    np.array(forcings_test[data_name])[591*2+6:-4:2],
                    np.array(forcings_test[data_name])[591*2+8:-2:2],
                    np.array(forcings_test[data_name])[591*2+9:-1:2],
                    np.array(forcings_test[data_name])[591*2+10::2],
                ]
            ).transpose((1, 0, 2, 3)),
    )

    #    norm = xr.merge([dataset_2009,dataset_2010,dataset_2011,
    #                    dataset_2012,dataset_2013,dataset_2014,
    #                   dataset_2015,dataset_2016])
 
        std = dataset_train[data_name].std(dim = ["time","prec","lat","lon"]).to_numpy()
        mean = dataset_train[data_name].mean(dim = ["time","prec","lat","lon"]).to_numpy()
        dataset_val[data_name] = (dataset_val[data_name] - mean) / std
        dataset_test[data_name] = (dataset_test[data_name] - mean) / std
        
        print(dataset_val[data_name].mean(skipna=True))
        print(dataset_test[data_name].std(skipna=True))
        print("WRITE")
        print(dataset_train)
        dataset_val.to_netcdf(path="./" + data_name + "_val_forcings.nc", mode="w")
        dataset_test.to_netcdf(path="./" + data_name + "_test_forcings.nc", mode="w")
    
    def create_sit_dataset(self, nextsim_data, variable, data_name):
        source = xr.open_dataset(nextsim_data)
        lat_source = source.variables["latitude"]
        lon_source = source.variables["longitude"]
        print("OPEN CS2SMOS DATASET")


        dataset_val = xr.open_mfdataset("CS2SMOS/2021/W_XX-ESA,SMOS_CS2,NH_25KM_EASE2_202*.nc")
        #dataset_val.fillna(0)
        print(dataset_val)
        time_val = dataset_val.time.data

        print(time_val)
        lat_target = dataset_val.lat
        lon_target = dataset_val.lon
        print(lat_target)
        print(np.shape(lon_target))
        #lon_target2d, lat_target2d = np.meshgrid(lon_target, lat_target)
        lon_target2d = lon_target.data
        lat_target2d = lat_target.data
        print("CONVERT LAT, LON TO CARTESIAN GRID")
        xt, yt, zt = self.lon_lat_to_cartesian(
            lon_source.values.flatten(), lat_source.values.flatten()
               )
        xs, ys, zs = self.lon_lat_to_cartesian(lon_target2d.flatten(), lat_target2d.flatten())

        print("INTERPOLATE")
        tree = cKDTree(np.column_stack((xs, ys, zs)))
        d, inds = tree.query(np.column_stack((xt, yt, zt)), k=1)

        
        forcings_val = self.interpolate(
            time_val, source, dataset_val, data_name, lat_source, d, inds
        )

        print("CREATE DATASETS")
        lat_source = lat_source[91:, 8:-8]
        lon_source = lon_source[91:, 8:-8]

        lat_source = lat_source[::self.N_res, ::self.N_res]
        lon_source = lon_source[::self.N_res, ::self.N_res]

        
        dataset_val = xr.Dataset(
            coords={
            "time": time_val[0:],
            "prec": [0],
            "lat": (["x", "y"], lat_source),
            "lon": (["x", "y"], lon_source),
            }
        ) 
        print("Add variables")
        
        dataset_val[data_name] = (
            ["time", "prec", "lat", "lon"],
            np.array(
                [
                    np.nan_to_num(np.array(forcings_val[data_name]), 0),
                ]
            ).transpose((1, 0, 2, 3)),
        )
        

    #    norm = xr.merge([dataset_2009,dataset_2010,dataset_2011,
    #                    dataset_2012,dataset_2013,dataset_2014,
    #                   dataset_2015,dataset_2016])
        mean = dataset_val[data_name].mean(skipna=True)
 
        std = dataset_val[data_name].std(skipna=True)


        mean = 0.38415005912623124
        std = 0.7710474897556033
        dataset_val[data_name] = (dataset_val[data_name] - mean) / std
        #dataset_test[data_name] = (dataset_test[data_name] - mean)/std
        np.save('mean_input.npy', mean)

        np.save('std_output.npy', std)
        
        print(dataset_val[data_name].mean(skipna=True))
        print("WRITE")
        print(dataset_val)
        dataset_val.to_netcdf(path="./sit_obs_2021.nc", mode="w")
        #dataset_test.to_netcdf(path="./sit_obs_test.nc", mode="w")
    def create_sit(self, path_to_data):
        self.create_sit_dataset(path_to_data, "2021_sit", "analysis_sea_ice_thickness")
    
    def create_forcings(self, path_to_data):
        self.create_forcings_dataset(path_to_data, "t2m", "t2m")
        self.create_forcings_dataset(path_to_data, "u10", "u10")
        self.create_forcings_dataset(path_to_data, "v10", "v10")


    def merge_sit_forcings(self):
    
        x1 = xr.open_dataset('sit_obs_2021.nc')
        
        x2 = xr.open_dataset('u10_test_forcings.nc')
        print(x2.time)
        x3 = xr.open_dataset('v10_test_forcings.nc')
        x5 = xr.open_dataset('t2m_test_forcings.nc')
        #x1["time"] = x2["time"]
        x2 = x2.sel(time=slice(x2['time'][0], x2['time'][177*2]))
        x3 = x3.sel(time=slice(x3['time'][0], x3['time'][177*2]))
        x5 = x5.sel(time=slice(x5['time'][0], x5['time'][177*2]))
        #x1["time"] = x2["time"]
        x = xr.merge([x1,x2,x3,x5])
        x.to_netcdf('2021_input.nc', mode = 'w')
        print(x)
        print(x.u10.data)
        #x1 = xr.open_dataset('sit_obs_test.nc')
        #print(x1.time)
        #x2 = xr.open_dataset('u10_test_forcings.nc')
        #print(x2.time)
        #x3 = xr.open_dataset('v10_test_forcings.nc')
        #x5 = xr.open_dataset('t2m_test_forcings.nc')
        #x1["time"] = x2["time"]
        #x2 = x2.sel(time=slice(x2['time'][1], x2['time'][100]))
        #x3 = x3.sel(time=slice(x3['time'][1], x3['time'][100]))
        #x5 = x5.sel(time=slice(x5['time'][1], x5['time'][100]))
        #x1["time"] = x2["time"]
        #x = xr.merge([x1,x2,x3,x5])
        #x.to_netcdf('test_input.nc', mode = 'w')
data = create_dataset_from_nextsim_outputs(N_res = 4)
#data.make_sit()

#data.create_forcings("../Data/Moorings_2018m08.nc")
#data.create_sit("../Data/Moorings_2018m08.nc")

data.merge_sit_forcings()
