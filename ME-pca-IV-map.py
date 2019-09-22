import ME
import numpy as np
import astropy.io.fits as fits
import os
import time
import sklearn.neighbors
import matplotlib.pyplot as plt


start = time.time()

directory = 'C:\\data\\hinode\\sp_20140510_105533\\hao\\web\\csac.hao.ucar.edu\\data\\hinode\\sot\\level1\\2014\\05\\10\\SP3D\\20140510_105533\\'
files_list = os.listdir(directory)
x = np.random.randint(len(files_list))


        
        
param_file = fits.open('C:\\data\\hinode\\20140510_105533.fits')


dims_names = ['Strength', 'Inclination', 'Azimuth', 'Doppler broadening', 'Damping', 'Line strength',
              'Source function', 'Source function gradient', 'Doppler shift', 'Filling factor', 'Stray shift']

dims_flags = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
dims = np.sum(dims_flags)

#constant parameters
wl0 = 6302.5
g = 2.5
mu = 1
l_v = [wl0, g, mu]

#linewidth argument
argument = np.linspace(6302.0692255, 6303.2544205, 56)
line_arg = 1000*(argument - wl0)

var_par = 11

#variable parameters spaces and reference
#[ default value, min, max]


B = [1000, 0, 2000]
theta = [0, 0, np.pi]
xi = [0, 0, np.pi]

D = [30, 20, 90]
gamma = [5, 5, 50]
etta_0 = [10, 10, 20]

S_0 = [0.5, 1, 1]
betta = [1, 1, 2]

Doppler_shift = [0, 0, 0]
filling_factor = [1, 0, 1]
stray_shift = [0, 0, 0]

param_def = np.transpose(np.array([B, theta, xi, D, gamma, etta_0, 
                      S_0, betta, Doppler_shift, filling_factor, stray_shift]))

nodes = 6

#making_spaces
spaces = []
print('Dimensions used - ', dims)
for i in range(var_par):
    if dims_flags[i]:
        print(dims_names[i])
        spaces.append(np.linspace(param_def[1][i], param_def[2][i], nodes) )

spaces = np.array(spaces)
steps = (spaces[::,-1] - spaces[::,-0])/(nodes - 1)
database = np.empty((nodes**dims, 56*6))

print('Total nodes - ', nodes**dims)
print('Space used - ', int(database.itemsize*database.size/1024/1024), 'MB')

#values = np.array(np.meshgrid(B_space, theta_space, xi_space, D_space, etta_0_space))
values = np.array(np.meshgrid(*spaces))
values = np.moveaxis(values, 0, -1)
values = np.reshape(values, (nodes**dims, dims))


#etta_0 differs!!!

for j in range(len(values)):
    p_v = []
    var_ctr = 0
    for par_c in range(var_par):          
        if dims_flags[par_c]:
            p_v.append(values[j][var_ctr])
            var_ctr += 1
        else:
            p_v.append(param_def[0][par_c])
    line = ME.ME_ff(l_v, p_v, line_arg)
    line = np.reshape(np.transpose(line), (56*4, 1)).flatten()
    I = line[0:56]
    Q = line[56:56*2]
    U = line[56*2:56*3]
    V = line[56*3:56*4]
    
    a = 0.5/(np.max(I) - np.min(I))
    b = 0.5*(np.max(I) - 2*np.min(I))/(np.max(I) - np.min(I))
    
    I = a*I + b
    Q = a*Q
    U = a*U
    V = a*V
    database[j] = np.concatenate((I + Q, I - Q, I + U, I - U, I + V, I - V))

base_time = time.time() - start
print(time.time() - start)

mean = np.mean(database, axis = 0)

database -= mean

cov_mat = np.cov(np.transpose(database))

pri_comp = 10            #number of used principal components

u, s, v = np.linalg.svd(cov_mat)
sp_rot = np.dot(database, u)
sp_rot = sp_rot[::, :pri_comp]

spectra_file = fits.open(directory + files_list[0])

x_len = len(files_list)
y_len = len(spectra_file[0].data[0])


inversion_map = np.empty((x_len, y_len, dims))

for x_c in range(x_len):
    spectra_file = fits.open(directory + files_list[x_c])
    print(x_c)
    for y_c in range(y_len):
        real_I = spectra_file[0].data[0][y_c][56:].astype('float64')*2
        real_Q = spectra_file[0].data[1][y_c][56:].astype('float64')
        real_U = spectra_file[0].data[2][y_c][56:].astype('float64')
        real_V = spectra_file[0].data[3][y_c][56:].astype('float64')
        
        a = 0.5/(np.max(real_I) - np.min(real_I))
        b = 0.5*(np.max(real_I) - 2*np.min(real_I))/(np.max(real_I) - np.min(real_I))
        
        real_I = a*(real_I) + b
        real_Q = a*real_Q
        real_U = a*real_U
        real_V = a*real_V
        
        real_sp = np.concatenate((real_I + real_Q, real_I - real_Q, real_I + real_U, real_I - real_U, real_I + real_V, real_I - real_V))
        real_sp -= mean
        real_pca = np.dot(real_sp, u)[:pri_comp]

        neigh = sklearn.neighbors.NearestNeighbors(n_neighbors = 3, algorithm = 'ball_tree')
        neigh.fit(sp_rot)
        closest = neigh.kneighbors(np.reshape(real_pca, (1,-1)))

        closest_params = values[closest[1][0][0]]
        
        inversion_map[x_c][y_c] = closest_params
 

neigh_time = time.time() - start - base_time       
print(time.time() - start - base_time)
im = np.swapaxes(inversion_map, 0, 2)
hdul = fits.HDUList([fits.PrimaryHDU(im)])
name = str(nodes) + '-' +  time.strftime("%H" + "%M", time.localtime())
hdul.writeto(name + '.fits')

