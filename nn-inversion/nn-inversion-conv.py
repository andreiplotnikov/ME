import sys
sys.path.append(r"..\\")
import os

import tensorflow as tf
import keras, keras.layers as L, keras.backend as K
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import astropy.io.fits as fits
import time

import ME
import MEbatch


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
    
    inp = L.Input((spectrum_shape,))
         
    net = L.BatchNormalization(axis = 1)(inp)
    net = L.Reshape( (int(spectrum_shape/4), 4) )(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(25, 5, activation = 'elu', padding = 'same')(net)
    net = L.MaxPool1D(2)(net)
    net = L.BatchNormalization(axis = 1)(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(15, 5, activation = 'elu', padding = 'same')(net)
    net = L.MaxPool1D(2)(net)
    net = L.BatchNormalization(axis = 1)(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    net = L.Conv1D(10, 2, activation = 'elu', padding = 'same')(net)
    encod = L.Flatten()(net)
    
    
    
    net = L.Concatenate()([encod, inp])
    out = L.Dense(parameters_shape*3)(net)
    out = L.Dense(parameters_shape)(net)
    
    model = keras.models.Model(inputs = inp, outputs = out)
  
    return model

def generate_profiles(line_vector, flags, spaces, argument, batch_size):
    param_vector = np.random.random((batch_size, len(flags)))*(spaces[2] - spaces[1]) + spaces[1]
    param_vector *= flags    
    param_vector += (1 - flags)*spaces[0]
        
    param_vector[:, 0] = 1000*(1 - np.log(np.e*np.random.rand(batch_size)))
            
    x = np.broadcast_to(argument, (batch_size, len(argument)))             
    
    if batch_size > 1: profile = MEbatch.ME_ff(line_vector, param_vector, x)
    else: profile = ME.ME_ff(line_vector, param_vector.flatten, argument)
    
    
    
            
    noise_level = 0.01*np.random.exponential(size = batch_size)
    noise = np.reshape(noise_level, (-1, 1))*np.random.randn(batch_size, 56*4)
    noise = np.reshape(noise, (batch_size, 56, 4))
    
    profile += noise
        
    param_vector[:, 8] += 50
    param_vector[:, 10] += 50
    
       
    return np.reshape(np.log1p(param_vector), (batch_size, 11) ), (np.reshape(np.swapaxes(profile, 1, 2), (-1, 4*56)))

class DataGenerator(keras.utils.Sequence):
    def __init__(self, line_vector, flags, spaces, argument, dim = (1, 56), batch_size = 10000, per_epoch = 100, n_channels = 1, n_classes = 10):
        self.dim = dim
        self.npe = per_epoch
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.line_vector = line_vector
        self.flags = flags
        self.spaces = spaces
        self.argument = argument
        self.batch_size = batch_size
    
    def __data_generation(self):
               
        param, prof = generate_profiles(self.line_vector, self.flags, self.spaces, self.argument, self.batch_size)
        
        return prof, param
    
    def __len__(self):
        return self.npe
    
    def __getitem__(self, index):
        X, Y = self.__data_generation()
        return X, Y
    
class learning_rate_drop(keras.callbacks.Callback):
    def __init__(self, value = 0.01, period = 50):
        super(learning_rate_drop, self).__init__()
        self.value = value
        self.period = period
        
    def on_epoch_end(self, epoch, logs = None):
        if (epoch + 1) % self.period == 0:
            print('Learning rate drop to value ', self.value)
            K.set_value(inversion.optimizer.lr, self.value)

class learning_rate_drop_threshold(keras.callbacks.Callback):
    def __init__(self, value = 0.01, threshold = 1e-4):
        super(learning_rate_drop_threshold, self).__init__()
        self.value = value
        self.threshold = threshold
        
    def on_epoch_end(self, epoch, logs = None):
        if K.get_value(inversion.optimizer.lr) < self.threshold:
            print('Learning rate drop to value ', self.value)
            K.set_value(inversion.optimizer.lr, self.value)      

reset_tf_session()
    
spectrum_shape  = 56*4

transitional_shape = 100

dims_names = ['Strength', 'Inclination', 'Azimuth', 'Doppler broadening', 'Damping', 'Line strength',
              'Continuum intensity', 'Source function gradient', 'Doppler shift', 'Filling factor', 'Stray shift']

dims_flags = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

wl0 = 6302.5
g = 2.5
mu = 1
l_v = [wl0, g, mu]

argument = np.linspace(6302.0692255, 6303.2544205, 56)
line_arg = 1000*(argument - wl0)

var_par = 11

B = [1000, 0, 5000]
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
                                   spaces = param_def, argument = line_arg, per_epoch = 30)

validation_generator = DataGenerator(line_vector = l_v, flags = dims_flags, 
                                   spaces = param_def, argument = line_arg, per_epoch = 3)

#dec_adamax = keras.optimizers.Adamax(decay = 1e-5)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.75,
                              verbose=1, patience=5, min_lr=0)

lrd = learning_rate_drop(value = 0.01, period = 50)

lrdt = learning_rate_drop_threshold(value = 0.01, threshold = 1e-5)


inversion.compile(optimizer = 'adamax', loss='mse')

history = inversion.fit_generator(generator = training_generator, 
                                    validation_data = validation_generator,
                                    epochs = 500, callbacks=[reduce_lr, lrd])


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()



def check_real(x_c, y_c, spaces):
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
    
    pred_params = np.expm1(pred_params)
    pred_params[:,8] -= 50
    pred_params[:,10] -= 50
    print(pred_params)
    
    pred_spectra = np.reshape(ME.ME_ff(l_v, pred_params, line_arg).T, (224, 1))
    
    pred_spectra /= np.max(pred_spectra)
    
    plot_spectrum(real_sp)
    plot_spectrum(pred_spectra)
    plt.show()
    
def params_from_real(x_c, y_c, spaces):
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
    
    pred_params = np.expm1(pred_params)
    pred_params[:,8] -= 50
    pred_params[:,10] -= 50
    
    print('predicted first: ', pred_params)
    
    pred_spectra = np.reshape(ME.ME_ff(l_v, pred_params, line_arg).T, (224, 1))
    
    pred_spectra /= np.max(pred_spectra)
    
    pred_params_2 = np.reshape(inversion.predict(np.reshape(pred_spectra, (1, 224))), (11))
    
    pred_params_2 = np.expm1(pred_params_2)
    pred_params_2[:,8] -= 50
    pred_params_2[:,10] -= 50
    
    print('predicted second: ', pred_params_2)
    
    pred_spectra_2 = np.reshape(ME.ME_ff(l_v, pred_params_2, line_arg).T, (224, 1))
    
    pred_spectra_2 /= np.max(pred_spectra_2)
    
    plot_spectrum(pred_spectra)
    plot_spectrum(pred_spectra_2)
    plt.show()
    
def save_model():
    localtime = time.localtime()
    name = '.\\models\\' + 'conv-' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]) + '.h5'
    inversion.save(name)
    
def continue_learning(epochs):
    history = inversion.fit_generator(generator = training_generator, 
                                    validation_data = validation_generator,
                                    epochs = epochs, callbacks=[reduce_lr])
    return history

def continue_learning_reset_lr(epochs, lr = 0.01):
    K.set_value(inversion.optimizer.lr, lr)
    history = inversion.fit_generator(generator = training_generator, 
                                    validation_data = validation_generator,
                                    epochs = epochs, callbacks=[reduce_lr])
    return history
       
