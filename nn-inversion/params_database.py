import sys


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import astropy.io.fits as fits
import time
import scipy.stats

import ME

directory = 'C:\\data\\hinode\\1\\'

a = os.listdir(directory)

files = list(fits.open(directory + a[i]) for i in range(len(a)))


parameters = np.empty((0, 11))

for i in range(len(files)):
    p = np.empty( (files[i][1].data.shape[0]*files[i][1].data.shape[1], 11))
    for j, k in enumerate([1, 2, 3, 6, 8, 7, 9, 10, 5, 12, 13]):
        p[:, j] = files[i][k].data.flatten()
    parameters = np.concatenate((parameters, p), axis = 0)

hdul = fits.HDUList([fits.PrimaryHDU(parameters)])
hdul.writeto('pb\\parameters_base1.fits', overwrite = 1) 
