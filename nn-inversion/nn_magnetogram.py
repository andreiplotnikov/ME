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

import ME
inversion = keras.models.load_model('.\\models\\2019_10_29_123725.h5')

directory = "D:\\fits\\data\\sp_20140926_170005\\hao\\web\\csac.hao.ucar.edu\\data\\hinode\\sot\\level1\\2014\\09\\26\\SP3D\\20140926_170005\\"
refer = fits.open("D:\\fits\\data\\20140926_170005.fits")

a = os.listdir(directory)
files_list = list(directory + a[i] for i in range(len(a)))

X_len = refer[0].header.get('NAXIS1')
Y_len = refer[0].header.get('NAXIS2')

inverted_params = np.empty( (X_len, Y_len, 11))


start = time.time()
for X_count in range(X_len):
    print(X_count)
    spectra_file = fits.open(files_list[X_count])
    for Y_count in range(Y_len):
        real_I = spectra_file[0].data[0][Y_count][56:].astype('float64')*2
        real_Q = spectra_file[0].data[1][Y_count][56:].astype('float64')
        real_U = spectra_file[0].data[2][Y_count][56:].astype('float64')
        real_V = spectra_file[0].data[3][Y_count][56:].astype('float64')
            
        real_sp = np.concatenate((real_I, real_Q, real_U, real_V))
        
        normalization = np.max(real_sp)
        real_sp /= normalization
        
        pred_params = np.reshape(inversion.predict(np.reshape(real_sp, (1, 224))), (11))
        
        pred_params[6] *= normalization
        
        inverted_params[X_count][Y_count] = pred_params
        
        
print(time.time() - start)
inverted_params = np.swapaxes(inverted_params, 0, 2)
