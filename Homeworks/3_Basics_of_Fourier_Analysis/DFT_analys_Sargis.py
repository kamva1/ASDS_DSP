
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

#--------- convert file -----------------------------------------
# ---command line------ ffmpeg -i 'aa.m4a' -ar 16000 'Sargis_aa.wav'
file_name = 'Sargis_aa_mono.wav'
sampling_rate, data = wavfile.read(file_name)
data = data.astype('float')
len_seconds = len(data) / sampling_rate
# print(len_seconds)
# exit()
N = len(data)

coef_no = int(N / 2) + 1 # amount of independent coefficients
samples = np.array(list(range(N))) # sample indices 
freqs = np.array(list(range(coef_no))) * sampling_rate / N # frequencies of current signal spectrum

coefs_Sargis = np.fft.rfft(data) # DFT coefs for 0, 1, ..., floor(N/2) base vectors
amplitude_spectr = np.abs(coefs_Sargis)
db_spectr = 10*np.log10(amplitude_spectr + 1) # db scale
k_max =  np.argmax(amplitude_spectr)
f_max = sampling_rate / N * k_max
print('maximal amplitude frequency: {}'.format(f_max))
exit()

#-------------- visualising amplitude spectrum -----------------
plt.plot(freqs, db_spectr)
plt.xlabel('freqs in hz')
plt.ylabel('amplituds')
plt.show()




