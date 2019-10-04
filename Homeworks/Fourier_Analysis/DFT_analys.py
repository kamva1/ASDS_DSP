
# DFT analyse task
#
# 	1) generate periodic triangular signals of various frequencies, and analyse amplitude spectrum
#
# 	2) pronounce any vowel sound(s) and record your voice, convert recording to 16 bit wav, and
# 	analyse spectrum of your voice (find main frequencies of your voice)
#
# below you can see the implementation of sinusoidal signal

import numpy as np
import scipy.io.wavfile as wavfile
import time
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
import glob

#--------- parameters -----------------------------------------
file_name = 'sinusoid.wav'
sampling_rate = 16000 # hz
omega = np.pi / 10 # radians per sample (pysical_frequency = sampling_rate * omega / (2*np.pi))
amplitude = 1000
N = 3 * sampling_rate # lenght of signal

coef_no = int(N / 2) + 1 # amount of independent coefficients
samples = np.array(list(range(N))) # sample indices 
freqs = np.array(list(range(coef_no))) * sampling_rate / N # frequencies of current signal spectrum

#--------generating signal ------------------------------------
sinusoid = amplitude*np.sin(omega * samples) # sinusoidal signal
sinusoid = sinusoid.astype('int16') 
wavfile.write(file_name, sampling_rate, sinusoid)
print(freqs)

coefs_sinusoid = np.fft.rfft(sinusoid) # DFT coefs for 0, 1, ..., floor(N/2) base vectors
amplitude_spectr = np.abs(coefs_sinusoid)
db_spectr = 10*np.log10(amplitude_spectr + 1)

#-------------- visualising amplitude spectrum -----------------
plt.plot(freqs, amplitude_spectr)
plt.xlabel('freqs in hz')
plt.ylabel('amplituds')
plt.show()




