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
import vector_normalization
import scipy.stats

import ME
inversion = keras.models.load_model('.\\models\\hs-2020_1_21_949.h5')

B = [1000, 0, 4000]
theta = [0, 0, np.pi]
xi = [0, 0, np.pi]

D = [30, 20, 90]
gamma = [5, 5, 50]
etta_0 = [10, 0.1, 100]

cont_int = [1, 0.5, 1.5]
betta = [1, 1, 2]

Doppler_shift = [0, -50, 50]
filling_factor = [1, 0, 1]
stray_shift = [0, -50, 50]

spaces = np.transpose(np.array([B, theta, xi, D, gamma, etta_0, 
                      cont_int, betta, Doppler_shift, filling_factor, stray_shift]))

directory = "D:\\fits\\data\\sp_20140926_170005\\hao\\web\\csac.hao.ucar.edu\\data\\hinode\\sot\\level1\\2014\\09\\26\\SP3D\\20140926_170005\\"
refer = fits.open("D:\\fits\\data\\20140926_170005.fits")

a = os.listdir(directory)
files_list = list(directory + a[i] for i in range(len(a)))

X_len = refer[0].header.get('NAXIS1')
Y_len = refer[0].header.get('NAXIS2')

inverted_params = np.empty( (X_len, Y_len, 11))


start = time.time()
for X_count in range(X_len):
    #print(X_count)
    spectra_file = fits.open(files_list[X_count])
    real_I = spectra_file[0].data[0][:,56:].astype('float64')*2
    real_Q = spectra_file[0].data[1][:,56:].astype('float64')*3
    real_U = spectra_file[0].data[2][:,56:].astype('float64')*3
    real_V = spectra_file[0].data[3][:,56:].astype('float64')*3
            
    real_sp = np.concatenate((real_I, real_Q, real_U, real_V), axis = 1)
        
    normalization = np.reshape(np.max(real_sp, axis = 1), (-1, 1))
    real_sp /= normalization
        
    pred_params = np.reshape(inversion.predict(np.reshape(real_sp, (-1, 224))), (-1, 11))
        
    pred_params = np.expm1(pred_params)
    pred_params[:,8] -= 10
    pred_params[:,10] -= 10
        
    pred_params[:, 6] *= normalization.flatten()
        
    inverted_params[X_count] = pred_params
        
        
print(time.time() - start)
inverted_params = np.swapaxes(inverted_params, 0, 2)

plt.plot(refer[1].data.flatten(), inverted_params[0].flatten(), ',')
plt.plot([0, 3000], [0, 3000])

print('Pearson\'s r:')
print('Field strength: ', scipy.stats.pearsonr(refer[1].data.flatten(), inverted_params[0].flatten())[0])
print('Field inclination: ', scipy.stats.pearsonr(refer[2].data.flatten(), (180/np.pi)*inverted_params[1].flatten())[0])
print('Field azimuth: ', scipy.stats.pearsonr(refer[3].data.flatten(), (180/np.pi)*inverted_params[2].flatten())[0])
print('Doppler broadening: ', scipy.stats.pearsonr(refer[6].data.flatten(), inverted_params[3].flatten())[0])
print('Damping: ', scipy.stats.pearsonr(refer[8].data.flatten(), inverted_params[4].flatten())[0])
print('Line strength: ', scipy.stats.pearsonr(refer[7].data.flatten(), inverted_params[5].flatten())[0])
print('Continuum intensity: ', scipy.stats.pearsonr(refer[32].data.flatten(), inverted_params[6].flatten())[0])
print('Source function gradient: ', scipy.stats.pearsonr(refer[10].data.flatten(), inverted_params[7].flatten())[0])
print('Doppler shift: ', scipy.stats.pearsonr(refer[5].data.flatten(), inverted_params[8].flatten())[0])
print('Filling factor: ', scipy.stats.pearsonr(refer[12].data.flatten(), inverted_params[9].flatten())[0])
print('Stray shift: ', scipy.stats.pearsonr(refer[13].data.flatten(), inverted_params[10].flatten())[0])

def show_plots():
    for i, j in enumerate([1, 2, 3, 6, 8, 7, 33, 10, 5, 12, 13]):
        plt.plot(refer[j].data.flatten(), inverted_params[i].flatten(), ',')
        #plt.plot([np.min(refer[j].data), np.max(refer[j].data)], [np.min(refer[j].data), np.max(refer[j].data)])
        plt.show()

