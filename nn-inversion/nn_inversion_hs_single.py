import sys
sys.path.append(r"D:\\ME-master\\ME-master")

import tensorflow as tf
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
import MEbatch
import MEbatch_hs

from shutil import copyfile

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
        
def build_inverse(parameters_shape = 1, spectrum_shape = 168):
    
    inp = L.Input((spectrum_shape,))
    net = L.Reshape( (int(spectrum_shape/3), 3) )(inp)   
    conv = L.Conv1D(filters = 64, kernel_size = 3, padding="same", activation="relu")(net)
    conv = L.Conv1D(filters = 64, kernel_size = 3, padding="same", activation="relu")(conv)
    conv = L.MaxPool1D(pool_size = 2)(conv)
    conv = L.Conv1D(filters = 128, kernel_size = 3, padding="same", activation="relu")(conv)
    conv = L.Conv1D(filters = 128, kernel_size = 3, padding="same", activation="relu")(conv)
    #conv = L.BatchNormalization()(conv)
    conv = L.MaxPool1D(pool_size = 2)(conv)
    conv = L.Conv1D(filters = 256, kernel_size = 3, padding="same", activation="relu")(conv)
    conv = L.Conv1D(filters = 256, kernel_size = 3, padding="same", activation="relu")(conv)
    #conv = L.BatchNormalization()(conv)
    flat = L.Flatten()(conv)
    compressed = L.Dense(1024, activation="elu")(flat)
    drop = L.Dropout(0.25)(compressed)
    compressed = L.Dense(1024, activation="elu")(drop)
    drop = L.Dropout(0.25)(compressed)
    output = L.Dense(1, activation = 'tanh')(drop) 
    model = keras.models.Model(inputs = inp, outputs = output) 
    return model
    

class DataGenerator_H(keras.utils.Sequence):
    def __init__(self, line_vector, argument, 
                 batch_size, base, list_IDs, target_parameter, shuffle = True):
        self.line_vector = line_vector
        self.argument = argument
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.base = base
        self.on_epoch_end()
        self.tp = target_parameter
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        
        X = np.empty( (self.batch_size, 11))
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = base[ID]
        
        x_arg = np.broadcast_to(self.argument, (self.batch_size, len(self.argument)))
        
        cont = X[:,6] + self.line_vector[2]*X[:,7]

        
        X[:,6] /= cont
        X[:,7] /= cont
        
        profile = MEbatch_hs.ME_ff(self.line_vector, X, x_arg)
        
        noise_level = np.array([109, 28, 28, 44])
        noise_level = np.broadcast_to(noise_level, (cont.shape[0], 4)).T/cont
        noise_level = np.reshape(noise_level.T, (-1, 1, 4))
        noise = noise_level*np.random.normal(size = profile.shape)
        
        profile += noise
        
        X[:,8] += 10
        X[:,10] += 10
        
        
        
        profile[:,:,1:3] = profile[:,:,1:3]
        profile[:,:,3] = profile[:,:,3]
        
        profile = profile[:, :, 1:]
        
        param, prof = (X[:,self.tp] - 2500)/5000, np.reshape(np.swapaxes(profile, 1, 2), (-1, 3*56))
        
        
        return prof, param
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        X, Y = self.__data_generation(list_IDs_temp)
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
    
    
spectrum_shape  = 56*3

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
net = build_inverse(spectrum_shape = 56*3, parameters_shape = 1)

print(net.summary())

es = EarlyStopping(monitor='val_loss', patience = 5, verbose=1)
inp = L.Input((spectrum_shape,))

out_parameters = net(inp)



base = fits.open('pb\\parameters_base.fits')[0].data


train, test = train_test_split(base, test_size = 0.1)

localtime = time.localtime()

try: 
    os.mkdir('.\\models\\' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]))
except: 
    pass

directory = '.\\models\\' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]) + '\\'


target_parameter = 0
    
inversion = keras.models.Model(inputs=inp, outputs=out_parameters)

training_generator = DataGenerator_H(line_vector = l_v, argument = line_arg, 
                batch_size = 256, base = train, 
                list_IDs = np.arange(train.shape[0]), target_parameter = target_parameter)

validation_generator = DataGenerator_H(line_vector = l_v, argument = line_arg, 
                batch_size = 256, base = test, 
                list_IDs = np.arange(test.shape[0]), target_parameter = target_parameter)

dec_adamax = keras.optimizers.Adamax(learning_rate = 1e-4, decay = 0.01, beta_1 = 0.9, beta_2 = 0.999)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.75,
                          verbose=1, patience=2, min_lr=0)

stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=3,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        )



lrd = learning_rate_drop(value = 0.01, period = 50)

inversion.compile(optimizer = 'adamax', loss='mae')

history = inversion.fit_generator(generator = training_generator, 
                                validation_data = validation_generator,
                                epochs = 50, callbacks=[reduce_lr, lrd])

param_range = np.max(base[:, target_parameter]) - np.min(base[:, target_parameter])

print( np.sqrt(history.history['val_loss'][-1]) / param_range)

inversion.save(directory + '%02d' % target_parameter + dims_names[target_parameter] + '.h5')



plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test') 
plt.legend()


    
def save_model():
    localtime = time.localtime()
    name = '.\\models\\' + 'hs-3-' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]) + '.h5'
    inversion.save(name)
    
