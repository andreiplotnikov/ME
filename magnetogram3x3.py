import sys
sys.path.append(r"D:\\ME-master\\ME-master")

import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
import keras, keras.layers as L, keras.backend as K
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import astropy.io.fits as fits
import time
import scipy.stats

from tqdm import tqdm

def make_patches(array, shape, location):
    if shape[0] < location[0] or shape[1] < location[1]:
        return None
    
    borders = ((array.shape[0] - location[0]) % shape[0], (array.shape[1] - location[1]) % shape[1])
    
    end = (array.shape[0] - borders[0], array.shape[1] - borders[1])
    
    subarray = array[location[0]:end[0], location[1]:end[1]].copy()
    
    rows = int(subarray.shape[0] / shape[0])
    cols = int(subarray.shape[1] / shape[1])
    
    
    patches = np.empty((0,) + shape)
    #locs = np.empty((0, 2))
    

#    for X_cell in range(rows):
#        for Y_cell in range(cols):
#            corner = (location[0] + X_cell*shape[0], location[1] + Y_cell*shape[1])           
#            locs = np.concatenate( (locs, np.array([corner])))

    a = np.split(subarray, rows, axis = 0)
    a = np.concatenate(a, axis = 1)
    patches = np.array(np.split(a, rows*cols, axis = 1))
    
    xv, yv = np.meshgrid( location[0] + shape[0]*np.arange(rows), location[1] + shape[1]*np.arange(cols), indexing = 'ij')
    xv = xv.flatten()
    yv = yv.flatten()
    #locs = np.array([xv, yv]).T
    
    return patches, (rows, cols)

def rebuild_from_patches(patches, target_shape, location, rc):
    array = np.empty(target_shape)
    array.fill(np.nan)
    
    rows, cols = rc
    
    a = np.concatenate(patches, axis = 1)
    a = np.split(a, rows, axis = 1)
    a = np.concatenate(a, axis = 0)
    
    end = np.array(location) + np.array(a.shape[:2])
    array[location[0]:end[0], location[1]:end[1]] = a
    
    return array

parameters_names = ['Strength', 'Inclination', 'Azimuth', 'Doppler broadening', 'Damping', 'Line strength',
              'Continuum intensity', 'Source function gradient', 'Doppler shift', 'Filling factor', 'Stray shift']            
            
patch_shape = (3, 3)    
    
models_directory = '.\\models3x3\\'
models_list = os.listdir(models_directory)

inversion = keras.models.load_model('.\\models3x3\\00Strength.h5')


directory = "D:\\fits\\data\\sp_20140926_170005\\hao\\web\\csac.hao.ucar.edu\\data\\hinode\\sot\\level1\\2014\\09\\26\\SP3D\\20140926_170005\\"
refer = fits.open("D:\\fits\\data\\20140926_170005.fits")

#directory = 'model_spectra_170005\\'
#refer = fits.open("D:\\ME-master\\ME-master\\inverted_parameters.fits")

a = os.listdir(directory)
files_list = list(directory + a[i] for i in range(len(a)))


Y_len, X_len = refer[1].data.shape 

full_spectra = np.empty((X_len, Y_len, 4*56))
normalization_map = np.empty((X_len, Y_len))

loc_inverted_params = np.empty( (np.prod(patch_shape), X_len, Y_len, 11))

inverted_params = np.empty( (X_len, Y_len, 11))

scaling = np.empty(X_len)

start = time.time()

for X_count in tqdm(range(X_len)):
    spectra_file = fits.open(files_list[X_count])
    
    scaling[X_count] = spectra_file[0].header.get('SPBSHFT') 
    
    real_I = spectra_file[0].data[0][:,56:].astype('float64')*2
    real_Q = spectra_file[0].data[1][:,56:].astype('float64')*3
    real_U = spectra_file[0].data[2][:,56:].astype('float64')*3
    real_V = spectra_file[0].data[3][:,56:].astype('float64')*3
            
    real_sp = np.concatenate((real_I, real_Q, real_U, real_V), axis = 1)
        
    normalization = np.reshape(np.max(real_sp, axis = 1), (-1, 1))
    real_sp /= normalization
    
    normalization_map[X_count] = normalization.flatten()
    
    full_spectra[X_count] = real_sp

for target_parameter in [3]:
    print(parameters_names[target_parameter])
    inversion = keras.models.load_model(models_directory + models_list[target_parameter])
    for loc in tqdm(range(np.prod(patch_shape))):
        X_loc = loc // patch_shape[0]
        Y_loc = loc % patch_shape[0] 
        
        location = (X_loc, Y_loc)
        
        patches, rc = make_patches(full_spectra, (3,3,224), location)
        i_patches = inversion.predict(patches)
        loc_inverted_params[loc, :, :, target_parameter] = rebuild_from_patches(i_patches, inverted_params[:, :, 0].shape, location, rc)
    K.clear_session()
    
inverted_params = np.nanmean(loc_inverted_params, axis = 0)
        
print('Time: ', time.time() - start)
inverted_params = np.swapaxes(inverted_params, 0, 2)

inverted_params[0] = inverted_params[0]*5000 + 2500
inverted_params[1] = inverted_params[1]*180
inverted_params[2] = inverted_params[2]*180
inverted_params[3] = inverted_params[3]*100
inverted_params[4] = inverted_params[4] + 1
inverted_params[5] = (inverted_params[5])*100 + 50

inverted_params = np.nan_to_num(inverted_params)

plt.plot(refer[1].data.flatten(), inverted_params[0].flatten(), ',')
plt.plot([0, 3000], [0, 3000])
plt.show()

Bz = inverted_params[0]*np.cos(np.radians(inverted_params[1]))
refer_Bz = refer[1].data * np.cos(np.radians(refer[2].data))

Bx = inverted_params[0]*np.sin(np.radians(inverted_params[1]))*np.cos(np.radians(inverted_params[2]))
refer_Bx = refer[1].data * np.sin(np.radians(refer[2].data)) * np.cos(np.radians(refer[3].data))

plt.plot(refer_Bz.flatten(), Bz.flatten(), ',')
plt.plot([0, 3000], [0, 3000])
plt.show()

plt.plot(refer_Bx.flatten(), Bx.flatten(), ',')
plt.plot([0, 3000], [0, 3000])
plt.show()


print('Pearson\'s r:')
print('Field strength: ', scipy.stats.pearsonr(refer[1].data.flatten(), inverted_params[0].flatten())[0])
print('Field inclination: ', scipy.stats.pearsonr(refer[2].data.flatten(), (180/np.pi)*inverted_params[1].flatten())[0])
print('Field azimuth: ', scipy.stats.pearsonr(refer[3].data.flatten(), (180/np.pi)*inverted_params[2].flatten())[0])
print('Doppler broadening: ', scipy.stats.pearsonr(refer[6].data.flatten(), inverted_params[3].flatten())[0])
print('Damping: ', scipy.stats.pearsonr(refer[8].data.flatten(), inverted_params[4].flatten())[0])
print('Line strength: ', scipy.stats.pearsonr(refer[7].data.flatten(), inverted_params[5].flatten())[0])
print('Continuum intensity: ', scipy.stats.pearsonr(refer[33].data.flatten(), inverted_params[6].flatten())[0])
print('Source function gradient: ', scipy.stats.pearsonr(refer[10].data.flatten(), inverted_params[7].flatten())[0])
print('Doppler shift: ', scipy.stats.pearsonr(refer[5].data.flatten(), inverted_params[8].flatten())[0])
print('Filling factor: ', scipy.stats.pearsonr(refer[12].data.flatten(), inverted_params[9].flatten())[0])
print('Stray shift: ', scipy.stats.pearsonr(refer[13].data.flatten(), inverted_params[10].flatten())[0])

def show_plots():
    for i, j in enumerate([1, 2, 3, 6, 8, 7, 33, 10, 5, 12, 13]):
        plt.plot(refer[j].data.flatten(), inverted_params[i].flatten(), ',')
        #plt.plot([np.min(refer[j].data), np.max(refer[j].data)], [np.min(refer[j].data), np.max(refer[j].data)])
        plt.show()
