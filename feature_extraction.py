import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.fft import fft
from scipy.signal.windows import hann
from scipy import signal
import matplotlib.pyplot as plt
import math
import os

def  block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs

    x = np.concatenate((x, np.zeros(blockSize)),axis=0)

    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb,t)

def compute_spectrogram(xb, fs):
    window = signal.hann(xb.shape[1])
    windowed_signal = xb*window
    X_jw = fft(windowed_signal)
    X = abs(X_jw)
    f = np.arange(X.shape[1])*fs/X.shape[1]
    bins = f.size // 2 + 1 
    return X[:,:bins].T, f[:bins]

def extract_spectral_centroid(xb, fs):
    X, f = compute_spectrogram(xb, fs)
    spec_centroid = np.sum(((X.T)*f),axis=1)/np.sum(X.T, axis=1)
    return spec_centroid

