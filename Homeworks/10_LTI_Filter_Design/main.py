import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as file
import os
import scipy.signal as signal


input_path = "_input"
output_path = "_output"
input_name = "test"
full_dir = input_path + "/" + input_name + ".wav"
low_freq = 500
hig_freq = 2000
order = 10

new_dir = os.path.dirname(full_dir)
sr, w = file.read(full_dir)
w = w.astype('float')
w_name = os.path.basename(full_dir).split('.wav')[0]
output_name_postfix = 'new'

def leaky_integrator(lamda, input):
	# single pole at lamda
	b = [1-lamda]
	a = [1, -lamda]
	output = signal.lfilter(b,a, input)
	return output, a, b

def DC_removal():
	# used to remove direct current or in other words value at f = 0
	# can be achieved taking one zero and one pole 
	return None

def resonator():
	# narrow bandpass filter
	# can be achieved using leaky integrator technique shifting the pole simmetrically
	# used to detect a sinusoid of current frequency
	return None

def hum_removal():
	# it is a type of notch filter, and can be used to remove value at fixed frequency 
	# can be achieved using pole and zeros symmetrical shifting
	return None


def bandpass_filtering(low_freq, hig_freq, order, input):
	# here low_freq and hig_freq are [0,1] segment numbers, the real freq is 
	# low_freq * max_freq, where max_freq = sample_rate / 2
    filter_coefs = signal.iirfilter(order, [low_freq, hig_freq], btype='bandpass', ftype='butter')
    b = filter_coefs[0];
    a = filter_coefs[1]
    output = signal.lfilter(b, a, input)
    return output, a, b

def lowpass_filtering(cutoff_freq, order, input):
	# here cutoff_freq is [0,1] segment number, the real freq is 
	# cutoff_freq * max_freq, where max_freq = sample_rate / 2
    filter_coefs = signal.iirfilter(order, cutoff_freq, btype = 'lowpass', ftype='butter')
    b = filter_coefs[0];
    a = filter_coefs[1]
    output = signal.lfilter(b, a, input)
    return output, a, b

def highpass_filtering(cutoff_freq, order, input):
	# here cutoff_freq is [0,1] segment number, the real freq is 
	# cutoff_freq * max_freq, where max_freq = sample_rate / 2
    filter_coefs = signal.iirfilter(order, cutoff_freq, btype = 'highpass', ftype='butter')
    b = filter_coefs[0];
    a = filter_coefs[1]
    output = signal.lfilter(b, a, input)
    return output, a, b

def random_filtering(input):
    b = np.array([1, 3 / 4 * np.random.rand() - 3 / 8, 3 / 4 * np.random.rand() - 3 / 8])
    a = np.array([1, 3 / 4 * np.random.rand() - 3 / 8, 3 / 4 * np.random.rand() - 3 / 8])
    output = signal.lfilter(b, a, input)
    return output, a, b

def plot_filter():
	return None


low_fname = str(int(low_freq / 1000)) + 'k'
hig_fname = str(int(hig_freq / 1000)) + 'k'

# obtaining frequencies as fractions
low_freq = low_freq * 2 / sr
hig_freq = hig_freq * 2 / sr


# filtering the input audio
# w_new, a, b = bandpass_filtering(low_freq, hig_freq, order, w)
w_new, a, b = highpass_filtering(hig_freq, order, w)
# w_new, a, b = random_filtering(w)
# w_new, a, b = leaky_integrator(0.99, w)
# plot_filter(a,b)


# new_name = w_name + '_' + output_name_postfix + '.wav'
# new_path = os.path.join(new_dir, new_name)
new_name = input_name + '_' + output_name_postfix + '.wav'
new_path = os.path.join(output_path, new_name)

file.write(new_path, sr, w_new.astype('int16'))
print('File with new append {} is ready'.format(output_name_postfix))
