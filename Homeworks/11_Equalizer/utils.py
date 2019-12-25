import numpy as np
import copy

from scipy import signal
import scipy.io.wavfile as wavfile

import time
import os

# import ffmpeg

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal

import glob


def convert_signal(path):
    """
    Convert m4a signal to wav signal
    
    Parameters:
        path (string): relative path of folder where m4a iles are located
        
    Returns:
        None - Function will add new wav file to path directory
    """

    for filename in os.listdir(path):
        if (filename.endswith(".m4a")) or (filename.endswith(".mp4")):
            os.system("ffmpeg -i {0} -ar 16000 {1}.wav".format(path + filename, path + filename[:-4]))
        else:
            continue

def read_audio(path, file_name):
    '''
    Read files from specified path (relative or absolute)
    
    Parameters:
    path (string): relative path to read file
    file_name (string): name of file located in path we want to read
    
    Returns:
    tuple: rate and date of wav file
    
    '''
    rate, data = wavfile.read(str(path) + str(file_name))
    # data, rate = librosa.load(str(path) + str(filename))
    data = data.astype('float')
    if np.mean(data**2) < 1:
        data = data * 2**15
    return rate, data

def write_audio(path, filename,  rate, data):
    '''
    Write files to specified path (relative or absolute) with volume transformation
    
    Parameters:
    path (string): relative path to write file
    file_name (string): name of file we want to save to located path
    rate (int): audio rate
    data (nd.array): the data we want to save
    volume (int): by default it settled 1, which means no transformation

    Returns:
    Boolean: If writing was finished successfully 
    
    '''
    data = copy.deepcopy(data)
    data = data.astype('int16')
    wavfile.write(str(path) + str(filename), rate, data)
    return True

def plot_waveform(data, start, end):
    '''
    Signal Visualization
    
    Parameters:
    data (nd.array): the data we want to visualize
    start (int): start range
    end (int): end range

    Returns:
    None: just shows the graph  
    
    '''
    data = data[start:end]
    plt.plot(data)
    plt.ylabel('amplitude')
    plt.xlabel('samples')
    plt.show()
    return None

