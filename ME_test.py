#import ME
import numpy as np
import scipy
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os
#import sunpy.wcs

import MEbatch_hs as ME
    
directory = 'C:\\data\\hinode\\sp_20140510_105533\\hao\\web\\csac.hao.ucar.edu\\data\\hinode\\sot\\level1\\2014\\05\\10\\SP3D\\20140510_105533\\'
files_list = os.listdir(directory)
param_file = fits.open('C:\\data\\hinode\\20140510_105533.fits')

x = 400
y = 300


spectra_file = fits.open(directory + str(files_list[x]))
spectra = np.array(list(spectra_file[0].data[i][y][56:] for i in range(4)))

spectra[0] *= 2

ref_length = spectra_file[0].header.get('CRVAL1')
ref_pixel = spectra_file[0].header.get('CRPIX1')
length_scale = spectra_file[0].header.get('CDELT1')
argument = list((ref_length+length_scale*(i+56-ref_pixel)) for i in range(len(spectra[0])))

X_angle = param_file[38].data[y][x]
Y_angle = param_file[39].data[y][x]

wl0 = 6302.5
g = 2.5
mu = 1

B = param_file[1].data[y][x]

theta = param_file[2].data[y][x]
xi = param_file[3].data[y][x]

D = param_file[6].data[y][x]
gamma = param_file[8].data[y][x]
etta_0 = param_file[7].data[y][x]

S_0 = param_file[9].data[y][x]
betta = param_file[10].data[y][x]

Dop_shift = param_file[5].data[y][x]*1e5*wl0*1e-8/3e10*1e8*1e3

filling_factor = param_file[12].data[y][x]
stray_shift = param_file[13].data[y][x]*1e5*wl0*1e-8/3e10*1e8*1e3

line_arg = 1000*(np.array(argument) - wl0)

cont_int = param_file[33].data[y][x]
profile = np.array(ME.ME_ff([wl0, g, mu], [B, theta, xi, D, gamma, etta_0, S_0, betta, Dop_shift, filling_factor, stray_shift], line_arg))

l_v = [wl0, g, mu]
p_i = [B, theta, xi, D, gamma, etta_0, S_0, betta, Dop_shift, filling_factor, stray_shift]

spectra_con = np.concatenate((spectra[0], spectra[1], spectra[2], spectra[3]))

profile = np.reshape(profile.T, (224))

#p = scipy.optimize.curve_fit(lambda x, *p_v: ME.ME_ff(l_v, p_v, x), line_arg, np.swapaxes(spectra, 0, 1), p0 = p_i, maxfev = 10000)[0]


p = scipy.optimize.leastsq(lambda p_v: np.sum(np.abs(np.reshape(ME.ME_ff(l_v, p_v, line_arg), (56, 4)) - np.swapaxes(spectra, 0, 1)), 1), x0 = p_i)[0]
I_opt, Q_opt, U_opt, V_opt = np.transpose(ME.ME_ff([wl0, g, mu], p , line_arg))


I, Q, U, V = np.transpose(ME.ME_ff([wl0, g, mu], p_i , line_arg))

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