import numpy as np
from skimage.measure import block_reduce

from tqdm import trange
import xarray as xr

from numpy.linalg import svd
import numpy.ma as ma


def prepare_data(sit):
    N_res = 4
    N = np.shape(sit)[0]
    x_t = []
    for i in trange(N):
        x_t.append(ma.getdata(sit[i]).reshape((603, 528, 1)))

    N_shape = int(512 / N_res)
    x_t = np.array(x_t)

    x_t = x_t[:, 91:, 8:-8]

    mask = ma.getmaskarray(sit[0])
    mask = mask.reshape((603, 528, 1))
    mask = mask[91:, 8:-8]

    x_t = np.where(mask == True, 0, np.array(x_t))

    if N_res > 1:
        x_coarse = np.zeros((np.shape(x_t)[0], N_shape, N_shape, 1))

    for i in trange(np.shape(x_t)[0]):
        x_coarse[i] = block_reduce(x_t[i], block_size=(N_res, N_res, 1), func=np.mean)

    x_t = x_coarse

    return x_t


# Open neXtSIM files
print("Start")

liste = []
years_train = ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]


for y in years_train:
    for m in months:
        liste.append("../../../data/source_files/Moorings_" + y + "m" + m + ".nc")
print(liste)
ds = xr.open_mfdataset(liste)
sit = ds.sit.to_masked_array()

x = prepare_data(sit)

mean_x = 0.38415005912623124
std_x = 0.7710474897556033
x = (x - mean_x) / std_x
print(x)
mask = np.load("../../../data/transfer/mask.npy")

mask = block_reduce(mask, block_size=(4, 4, 1), func=np.min)
mask = 1 - mask

mask_flatten = mask.flatten()
unflatten = np.where(mask_flatten > 0)
N_z_reduced = np.shape(unflatten[0])[0]

x_1D = x.reshape((-1, 128 * 128))[:, unflatten[0]]

print(x_1D)
X = x_1D - x_1D.mean(axis=0).reshape((1, N_z_reduced))
mean = x_1D.mean(axis=0).reshape((1, N_z_reduced))
np.save("mean_norm.npy", mean)
print(np.shape(X))
U, S, Vh = svd(X)

eigen_values = np.diag(S ** 2)
np.save("EOFs_y8_norm.npy", Vh.T)
np.save("U_y8.npy", U)
np.save("S_y8.npy", S)
# np.save('x_train.npy', x)
