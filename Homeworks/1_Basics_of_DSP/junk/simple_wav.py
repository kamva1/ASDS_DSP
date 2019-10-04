# sampling rate = 16000 , points per second
# 1) compute power of this wav
# 2) create function which plots previously defined segment of data
# 3) study signal data (print various segments of signal)
# 4) write new wav with 2 times greater volume

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.io
import itertools
import math
import time
#import adaptfilt as adf
import os
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
import glob

file_path = 'v5.wav'
start = 16000
end = 24000

def read_audio(path):
	sr, data = wavfile.read(path)
	data = data.astype('float32')
	return data

def write_audio(path, sr, data):
	data = data.astype('int16')
	wavfile.write(path, sr, data)
	return None

def compute_power(data):
	power = np.mean(data**2)
	return power

def plot_waveform(data):
	plt.plot(data)
	plt.ylabel('amplitude')
	plt.xlabel('samples')
	plt.show()
	return None

audio_data = read_audio(file_path)
plot_waveform(audio_data[start:end])
