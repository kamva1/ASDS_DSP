import numpy as np
import time
from os import path
import os
import importlib
import sys
from scipy import signal
import scipy.io.wavfile as file
import glob
import math
import matplotlib.pyplot as plt

class OLA:
    def __init__(self):

        self.sample_rate = 16000
        self.frame_dur = 32 # ms
        self.frame_len = self.frame_dur * self.sample_rate // 1000
        self.coef_no = self.frame_len // 2 + 1
        self.shift = self.frame_len // 2

    def lin_resp(self, n):
        resp = np.array([i/n for i in range(n)])
        return resp

    def triang_resp(self, n):
        x = np.zeros(n)
        x[0:n//2] = np.array([i/n for i in range(n//2)])
        x[n//2: 2*(n//2)] = 1/2 - x[0:n//2]
        return x

    def lowpass_resp(self, n):
        x = np.zeros(n)
        x[0:n//2] = 1
        x[n//2:] = 0
        return x


    def test(self, in_data, resp):
        # in_data : input data
        # resp: amplitude response of filter
    
        i = int(0)
        all_return_data = np.zeros(len(in_data))

        start_time = time.time()
        l = len(in_data)

        # here you can take any custom window function
        window = np.hanning(self.frame_len)

        while i < l - self.frame_len:
            frame = np.array(in_data[i:i + self.frame_len])
            # applying window function
            frame_w = frame * window
            coefs = np.fft.rfft(frame_w)
            # obtain coefs of filtered frame
            new_ceofs = resp * coefs
            new_frame = np.fft.irfft(new_ceofs)
            all_return_data[i:i + self.frame_len] = all_return_data[i:i + self.frame_len] + new_frame

            i += self.shift

        processed_time = time.time() - start_time
        return all_return_data, processed_time
