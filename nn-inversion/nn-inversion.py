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

def reset_tf_session():
    K.clear_session()
    tf.reset_default_graph()
    s = K.get_session()
    return s

def plot_spectrum(profile):
  
    plt.title('sample spectrum')
  
    
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(profile[i*56:(i+1)*56])
        
def build_inverse(spectrum_shape, transitional_shape, parameters_shape):
    """
    Here we define a simple linear autoencoder as described above.
    We also flatten and un-flatten data to be compatible with image shapes
    """
    
    net = keras.models.Sequential()
    net.add(L.InputLayer((spectrum_shape,)))
    #encoder.add(L.Flatten())                  #flatten image to vector
    net.add(L.Dense(transitional_shape, activation = 'relu'))           #actual encoder
    net.add(L.Dense(parameters_shape))
       
    return net

def generate_profiles(line_vector, flags, spaces, argument):
    param_vector = np.empty(len(flags))
    for j in range(len(flags)):
        if (flags[j]):
            param_vector[j] = spaces[1][j] + (spaces[2][j] - spaces[1][j])*np.random.random()
        else:
            param_vector[j] = spaces[0][j]
    profile = ME.ME_ff(line_vector, param_vector, argument)
        
    noise_level = 0.05 + 0.15*np.random.rand()
    noise = noise_level*np.random.randn(56, 4)
    
    profile += noise
    
    #weights = np.concatenate( (np.full(56, 1), np.full(56*3,3))  )
    weights = np.concatenate( (np.full(56, 1), np.full(56*3,1))  )
    
    return np.reshape(param_vector, (1, 11) ), (np.reshape(profile.T, (1, 56*4))*weights)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, line_vector, flags, spaces, argument, dim = (1, 56), per_epoch = 100, n_channels = 1, n_classes = 10):
        self.dim = dim
        self.npe = per_epoch
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.line_vector = line_vector
        self.flags = flags
        self.spaces = spaces
        self.argument = argument
    
    def __data_generation(self):
        param, prof = generate_profiles(self.line_vector, self.flags, self.spaces, self.argument)
        return prof, param
    
    def __len__(self):
        return self.npe
    
    def __getitem__(self, index):
        X, Y = self.__data_generation()
        return X, Y
    
spectrum_shape  = 56*4

transitional_shape = 100

dims_names = ['Strength', 'Inclination', 'Azimuth', 'Doppler broadening', 'Damping', 'Line strength',
              'Continuum intensity', 'Source function gradient', 'Doppler shift', 'Filling factor', 'Stray shift']

dims_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

wl0 = 6302.5
g = 2.5
mu = 1
l_v = [wl0, g, mu]

argument = np.linspace(6302.0692255, 6303.2544205, 56)
line_arg = 1000*(argument - wl0)

var_par = 11

B = [1000, 0, 4000]
theta = [0, 0, np.pi]
xi = [0, 0, np.pi]

D = [30, 20, 90]
gamma = [5, 5, 50]
etta_0 = [10, 2, 20]

cont_int = [1, 0.5, 1.5]
betta = [1, 1, 2]

Doppler_shift = [0, 0, 0]
filling_factor = [1, 0, 1]
stray_shift = [0, 0, 0]

param_def = np.transpose(np.array([B, theta, xi, D, gamma, etta_0, 
                      cont_int, betta, Doppler_shift, filling_factor, stray_shift]))

net = build_inverse(spectrum_shape, transitional_shape, len(dims_flags))

print(net.summary())

weights = np.concatenate( (np.full(56, 1), np.full(56*3,3))  )

es = EarlyStopping(monitor='val_loss', patience = 5, verbose=1)
inp = L.Input((spectrum_shape,))

out_parameters = net(inp)

inversion = keras.models.Model(inputs=inp, outputs=out_parameters)

training_generator = DataGenerator(line_vector = l_v, flags = dims_flags, 
                                   spaces = param_def, argument = line_arg, per_epoch = 50000)

validation_generator = DataGenerator(line_vector = l_v, flags = dims_flags, 
                                   spaces = param_def, argument = line_arg, per_epoch = 10000)

inversion.compile(optimizer='adamax', loss='mse')

history = inversion.fit_generator(generator = training_generator, 
                                    validation_data = validation_generator,
                                    epochs = 10)

def check_real(x_c, y_c):
    directory = 'D:\\fits\\hao\\web\\csac.hao.ucar.edu\\data\\hinode\\sot\\level1\\2017\\09\\05\\SP3D\\20170905_030404\\'
    files_list = os.listdir(directory)
    spectra_file = fits.open(directory + files_list[x_c])
    real_I = spectra_file[0].data[0][y_c][56:].astype('float64')*2
    real_Q = spectra_file[0].data[1][y_c][56:].astype('float64')
    real_U = spectra_file[0].data[2][y_c][56:].astype('float64')
    real_V = spectra_file[0].data[3][y_c][56:].astype('float64')
            
                
    real_sp = np.concatenate((real_I, real_Q, real_U, real_V))
    
    normalization = np.max(real_sp)
    
    real_sp /= normalization
    
    pred_params = np.reshape(inversion.predict(np.reshape(real_sp, (1, 224))), (11))
    print(pred_params)
    
    pred_spectra = np.reshape(ME.ME_ff(l_v, pred_params, line_arg).T, (224, 1))
    
    pred_spectra /= np.max(pred_spectra)
    
    plot_spectrum(real_sp)
    plot_spectrum(pred_spectra)
    plt.show()
    
def params_from_real(x_c, y_c):
    directory = 'D:\\fits\\hao\\web\\csac.hao.ucar.edu\\data\\hinode\\sot\\level1\\2017\\09\\05\\SP3D\\20170905_030404\\'
    files_list = os.listdir(directory)
    spectra_file = fits.open(directory + files_list[x_c])
    real_I = spectra_file[0].data[0][y_c][56:].astype('float64')*2
    real_Q = spectra_file[0].data[1][y_c][56:].astype('float64')
    real_U = spectra_file[0].data[2][y_c][56:].astype('float64')
    real_V = spectra_file[0].data[3][y_c][56:].astype('float64')
                
    real_sp = np.concatenate((real_I, real_Q, real_U, real_V))
    
    real_sp /= np.max(real_sp)
    
    pred_params = np.reshape(inversion.predict(np.reshape(real_sp, (1, 224))), (11))
    
    print('predicted first: ', pred_params)
    
    pred_spectra = np.reshape(ME.ME_ff(l_v, pred_params, line_arg).T, (224, 1))
    
    pred_spectra /= np.max(pred_spectra)
    
    pred_params_2 = np.reshape(inversion.predict(np.reshape(pred_spectra, (1, 224))), (11))
    
    print('predicted second: ', pred_params_2)
    
    pred_spectra_2 = np.reshape(ME.ME_ff(l_v, pred_params_2, line_arg).T, (224, 1))
    
    pred_spectra_2 /= np.max(pred_spectra_2)
    
    plot_spectrum(pred_spectra)
    plot_spectrum(pred_spectra_2)
    plt.show()
    
def save_model():
    localtime = time.localtime()
    name = '.\\models\\' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]) + '.h5'
    inversion.save(name)