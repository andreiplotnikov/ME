import ME
import numpy as np
import scipy
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os
import sunpy.wcs
import MEbatch_hs
import time

    
directory = "D:\\fits\\data\\sp_20140926_170005\\hao\\web\\csac.hao.ucar.edu\\data\\hinode\\sot\\level1\\2014\\09\\26\\SP3D\\20140926_170005\\"
param_file = fits.open("D:\\fits\\data\\20140926_170005.fits")
files_list = os.listdir(directory)

Y_len, X_len = (param_file[1].data.shape)

Y_len, X_len = (200, 100)

params = np.empty( (X_len, Y_len, 11))

start = time.time()

for X_c in range(X_len):
    spectra_file = fits.open(directory + str(files_list[X_c]))
    print(X_c, time.time() - start)
    for Y_c in range(Y_len):
        
        spectra = np.array(list(spectra_file[0].data[i][Y_c][56:] for i in range(4))).astype('float64')
        
        spectra[0] = spectra[0]*2
        
        ref_length = spectra_file[0].header.get('CRVAL1')
        ref_pixel = spectra_file[0].header.get('CRPIX1')
        length_scale = spectra_file[0].header.get('CDELT1')
        argument = list((ref_length+length_scale*(i+56-ref_pixel)) for i in range(len(spectra[0])))
        
        X_angle = param_file[38].data[Y_c][X_c]
        Y_angle = param_file[39].data[Y_c][X_c]
        
        wl0 = 6302.5
        g = 2.5
        mu = 1
        
        B = param_file[1].data[Y_c][X_c]
        
        theta = param_file[2].data[Y_c][X_c]
        xi = param_file[3].data[Y_c][X_c]
        
        D = param_file[6].data[Y_c][X_c]
        gamma = param_file[8].data[Y_c][X_c]
        etta_0 = param_file[7].data[Y_c][X_c]
        
        
        S_0 = param_file[9].data[Y_c][X_c]
        betta = param_file[10].data[Y_c][X_c]
        
        Dop_shift = param_file[5].data[Y_c][X_c]
        
        filling_factor = param_file[12].data[Y_c][X_c]
        stray_shift = param_file[13].data[Y_c][X_c]
        
        line_arg = 1000*(np.array(argument) - wl0)
        
        
                
        l_v = [wl0, g, mu]
        p_i = np.array([B, theta, xi, D, gamma, etta_0, S_0, betta, Dop_shift, filling_factor, stray_shift])
        
        I, Q, U, V = MEbatch_hs.ME_ff([wl0, g, mu], p_i, line_arg)[0].T
        
        #p_i = [500, 0, 0, 30, 20, 10, 20000, 2, 0, 1, 0]
        
        
        spectra_con = np.concatenate((spectra[0], spectra[1], spectra[2], spectra[3]))
                
        #p = scipy.optimize.curve_fit(lambda x, *p_v: ME.ME_ff(l_v, p_v, x), line_arg, np.swapaxes(spectra, 0, 1), p0 = p_i, maxfev = 10000)[0]
        
        weights = np.array([1, 3, 3, 3])
        
        p = scipy.optimize.leastsq(lambda p_v: np.sum(np.abs(MEbatch_hs.ME_ff(l_v, p_v, line_arg)[0]*weights - np.swapaxes(spectra, 0, 1)*weights), 1), x0 = p_i)[0]
        
        params[X_c][Y_c] = p
        
        I_opt, Q_opt, U_opt, V_opt = np.transpose(MEbatch_hs.ME_ff([wl0, g, mu], p , line_arg))

params = np.swapaxes(np.swapaxes(params, 0, 2), 1, 2)

line = np.array([I, Q, U, V])

plt.subplot(221)
plt.plot(argument, spectra[0])
plt.plot(argument, I, linestyle='dashed')
plt.plot(argument, I_opt)

plt.subplot(222)
plt.plot(argument, spectra[1])
plt.plot(argument, Q, linestyle='dashed')
plt.plot(argument, Q_opt)

plt.subplot(223)
plt.plot(argument, spectra[2])
plt.plot(argument, U, linestyle='dashed')
plt.plot(argument, U_opt)

plt.subplot(224)
plt.plot(argument, spectra[3])
plt.plot(argument, V, linestyle='dashed')
plt.plot(argument, V_opt)

plt.show()

#print(S_0, betta)