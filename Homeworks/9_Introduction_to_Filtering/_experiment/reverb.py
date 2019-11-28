# here is the simple example of applying room impulse response to the given audio
from os import path
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

input_audio_path = 'v1.wav'
output_audio_path = 'v1_rir.wav'
rir_path = 'rir.wav'

def rir_filter(w_data, rir_data):
	# this function applies reverberation filter to data
	# w_data: input data
	# rir_data: impulse response of reverberation filter or shortly room impulse response 
    l = len(w_data)
    max_ampl = np.max(np.abs(w_data))
    conv_data = signal.fftconvolve(w_data, rir_data)
    conv_data = max_ampl / np.max(np.abs(conv_data)) * conv_data # scaling output to have the same max as input
    return conv_data

def read_wav(in_path):
	sr, data = wavfile.read(in_path)
	data = data.astype('float')
	return sr, data

sr, audio = read_wav(input_audio_path)
sr, rir = read_wav(rir_path)
audio_reverb = rir_filter(audio, rir)
wavfile.write(output_audio_path, sr, audio_reverb.astype('int16'))