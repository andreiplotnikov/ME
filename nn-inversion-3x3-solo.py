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
        
def build_inverse(parameters_shape = 1, spectrum_shape = (3, 3, 56*4, )):
    
    inp = L.Input((spectrum_shape))
    conv = L.Conv2D(filters = 64, kernel_size = (2, 2), padding="same", activation="relu")(inp)
    conv = L.Conv2D(filters = 64, kernel_size = (2, 2), padding="same", activation="relu")(conv)
    conv = L.Conv2D(filters = 64, kernel_size = (2, 2), padding="same", activation="relu")(conv)
    conv = L.Conv2D(filters = 64, kernel_size = (2, 2), padding="same", activation="relu")(conv)
    #conv = L.MaxPool1D(pool_size = 2)(conv)
    conv = L.Conv2D(filters = 128, kernel_size = (2, 2), padding="same", activation="relu")(conv)
    conv = L.Conv2D(filters = 128, kernel_size = (2, 2), padding="same", activation="relu")(conv)
    conv = L.Conv2D(filters = 128, kernel_size = (2, 2), padding="same", activation="relu")(conv)
    conv = L.Conv2D(filters = 128, kernel_size = (2, 2), padding="same", activation="relu")(conv)
    #conv = L.BatchNormalization()(conv)
    #conv = L.MaxPool1D(pool_size = 2)(conv)
    conv = L.Conv2D(filters = 256, kernel_size = (2, 2), padding="same", activation="relu")(conv)
    conv = L.Conv2D(filters = 256, kernel_size = (2, 2), padding="same", activation="relu")(conv)
    conv = L.Conv2D(filters = 256, kernel_size = (2, 2), padding="same", activation="relu")(conv)
    conv = L.Conv2D(filters = 256, kernel_size = (2, 2), padding="same", activation="relu")(conv)
    #conv = L.BatchNormalization()(conv)
    flat = L.Flatten()(conv)
    compressed = L.Dense(1024, activation="elu")(flat)
    drop = L.Dropout(0.5)(compressed)
    compressed = L.Dense(1024, activation="elu")(drop)
    drop = L.Dropout(0.5)(compressed)
    output = L.Dense(9, activation = 'tanh')(drop) 
    output = L.Reshape((3, 3))(output)
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
        
        X = np.empty( (self.batch_size, 3, 3, 11))
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = base[ID]
        
        x_arg = np.broadcast_to(self.argument, (9*self.batch_size, len(self.argument)))
        
        cont = X[:,:,:, 6] + self.line_vector[2]*X[:,:,:,7]

        
        X[:,:,:, 6] /= cont
        X[:,:,:, 7] /= cont
        
        profile = MEbatch_hs.ME_ff(self.line_vector, np.reshape(X, (-1, 11)), x_arg)
        
        
        noise_level = np.array([109, 28, 28, 44])
        
        noise_level = np.broadcast_to(noise_level, (profile.shape[0], 4)).T/np.reshape(cont, (-1, 1)).T
        noise_level = np.reshape(noise_level.T, (-1, 1, 4))
        noise = noise_level*np.random.normal(size = profile.shape)
        
        profile += noise
        
        #X[:,8] += 10
        #X[:,10] += 10
        
        
        
        profile[:,:,1:3] = profile[:,:,1:3]*3
        profile[:,:,3] = profile[:,:,3]*3

        #profile = profile[:, :, 1:]
        
        #param, prof = np.log1p(X), np.reshape(np.swapaxes(profile, 1, 2), (-1, 4*56))
        
        profile = np.reshape(np.swapaxes(profile, 1, 2), (-1, 4*56))
        
        normalization = np.reshape(np.max(profile, axis = 1), (-1, 1))        
        profile /= normalization
        
        profile = np.reshape(profile, (-1, 3, 3, 4*56))

        param, prof = (X[:,:,:,self.tp])/100, profile
        
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
    
    
spectrum_shape  = (3, 3, 56*4, )

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
net = build_inverse(spectrum_shape = (3, 3, 56*4, ), parameters_shape = 1)

print(net.summary())

es = EarlyStopping(monitor='val_loss', patience = 5, verbose=1)
inp = L.Input((spectrum_shape))

out_parameters = net(inp)



base = fits.open('pb\\parameters_base3x3.fits')[0].data

base = np.moveaxis(base, 1, 3)


train, test = train_test_split(base, test_size = 0.1)

localtime = time.localtime()

try: 
    os.mkdir('.\\models\\' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]))
except: 
    pass

directory = '.\\models\\' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]) + '\\'



#for target_parameter in range(len(dims_flags)):
for target_parameter in [3]:
    
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



    lrd = learning_rate_drop(value = 0.001, period = 30)

    #inversion.compile(optimizer = dec_adamax, loss= 'mae')
    inversion.compile(optimizer = dec_adamax, loss= 'mean_absolute_percentage_error')

    history = inversion.fit_generator(generator = training_generator, 
                                    validation_data = validation_generator,
                                    epochs = 150, callbacks=[reduce_lr, lrd])
    
    param_range = np.max(base[:, target_parameter]) - np.min(base[:, target_parameter])
    
    print( np.sqrt(history.history['val_loss'][-1]) / param_range)
    
    inversion.save(directory + '%02d' % target_parameter + dims_names[target_parameter] + '.h5')



plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test') 
plt.legend()



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
    
    pred_params = np.expm1(pred_params)
    pred_params[8] -= 10
    pred_params[10] -= 10
    print(pred_params)
    
    pred_spectra = np.reshape(MEbatch_hs.ME_ff(l_v, pred_params, line_arg).T, (224, 1))
    
    pred_spectra /= np.max(pred_spectra)
    
    plot_spectrum(real_sp)
    plot_spectrum(pred_spectra)
    plt.show()
    
def params_from_real(x_c, y_c):
    directory = 'D:\\fits\\hao\\web\\csac.hao.ucar.edu\\data\\hinode\\sot\\level1\\2017\\09\\05\\SP3D\\20170905_030404\\'
    files_list = os.listdir(directory)
    spectra_file = fits.open(directory + files_list[x_c])
    real_I = spectra_file[0].data[0][y_c][56:].astype('float64')*2
    real_Q = spectra_file[0].data[1][y_c][56:].astype('float64')*10
    real_U = spectra_file[0].data[2][y_c][56:].astype('float64')*10
    real_V = spectra_file[0].data[3][y_c][56:].astype('float64')*5
                
    real_sp = np.concatenate((real_I, real_Q, real_U, real_V))
    
    real_sp /= np.max(real_sp)
    
    pred_params = np.reshape(inversion.predict(np.reshape(real_sp, (1, 224))), (11))
    
    pred_params = np.expm1(pred_params)
    pred_params[8] -= 10
    pred_params[10] -= 10
    
    print('predicted first: ', pred_params)
    
    pred_spectra = np.reshape(MEbatch_hs.ME_ff(l_v, pred_params, line_arg).T, (224, 1))
    
    pred_spectra /= np.max(pred_spectra)
    
    pred_params_2 = np.reshape(inversion.predict(np.reshape(pred_spectra, (1, 224))), (11))
    
    pred_params_2 = np.expm1(pred_params_2)
    pred_params_2[8] -= 10
    pred_params_2[10] -= 10
    
    print('predicted second: ', pred_params_2)
    
    pred_spectra_2 = np.reshape(MEbatch_hs.ME_ff(l_v, pred_params_2, line_arg).T, (224, 1))
    
    pred_spectra_2 /= np.max(pred_spectra_2)
    
    plot_spectrum(pred_spectra)
    plot_spectrum(pred_spectra_2)
    plt.show()
    
def save_model():
    localtime = time.localtime()
    name = '.\\models\\' + 'hs-3-' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]) + '.h5'
    inversion.save(name)
    
def continue_learning(directory, epochs):
    models_list = os.listdir(directory)
    
    localtime = time.localtime()
    try: 
        os.mkdir('.\\models\\' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]))
    except: 
        pass
    
    save_directory = '.\\models\\' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]) + '\\'
    for tp in range(11): 
        inversion = keras.models.load_model(directory + models_list[tp])
        
        training_generator = DataGenerator_H(line_vector = l_v, argument = line_arg, 
                    batch_size = 30000, base = train, 
                    list_IDs = np.arange(train.shape[0]), target_parameter = tp)

        validation_generator = DataGenerator_H(line_vector = l_v, argument = line_arg, 
                    batch_size = 30000, base = test, 
                    list_IDs = np.arange(test.shape[0]), target_parameter = tp)

        
        history = inversion.fit_generator(generator = training_generator, 
                                            validation_data = validation_generator,
                                            epochs = epochs, callbacks=[reduce_lr])
        param_range = np.max(base[:, tp]) - np.min(base[:, tp])

        print(tp,  np.sqrt(history.history['val_loss'][-1]) / param_range)
        inversion.save(save_directory + '%02d' % tp + dims_names[tp] + '.h5')

        
    return history, save_directory

def continue_learning_single(directory, epochs, tp):
    models_list = os.listdir(directory)
    
    localtime = time.localtime()
    try: 
        os.mkdir('.\\models\\' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]))
    except: 
        pass
    

    save_directory = '.\\models\\' + str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4]) + '\\'
    
    inversion = keras.models.load_model(directory + models_list[tp])   
    for j in range(11):
        if j != tp:
            copyfile(directory + models_list[j], save_directory + models_list[j])
            
    training_generator = DataGenerator_H(line_vector = l_v, argument = line_arg, 
                batch_size = 30000, base = train, 
                list_IDs = np.arange(train.shape[0]), target_parameter = tp)

    validation_generator = DataGenerator_H(line_vector = l_v, argument = line_arg, 
                batch_size = 30000, base = test, 
                list_IDs = np.arange(test.shape[0]), target_parameter = tp)

    
    history = inversion.fit_generator(generator = training_generator, 
                                        validation_data = validation_generator,
                                        epochs = epochs, callbacks=[reduce_lr, lrd])
    param_range = np.max(base[:, tp]) - np.min(base[:, tp])

    print(tp,  np.sqrt(history.history['val_loss'][-1]) / param_range)
    inversion.save(save_directory + '%02d' % tp + dims_names[tp] + '.h5')
    

    
    return history, save_directory

def continue_learning_reset_lr(epochs, lr = 0.01):
    K.set_value(inversion.optimizer.lr, lr)
    history = inversion.fit_generator(generator = training_generator, 
                                    validation_data = validation_generator,
                                    epochs = epochs, callbacks=[reduce_lr])
    return history
    
    