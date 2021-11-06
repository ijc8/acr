import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
import scipy.signal

from sklearn.neighbors import KNeighborsClassifier

import eval


def filter_transients(x, n):
    return np.min(x[:len(x)//n*n].reshape((-1, n)), axis=1)

def find_strokes(x, verbose=False):
    X = mlab.specgram(x, NFFT=512, noverlap=256)[0]
    summed = np.sum(X[15:30], axis=0)
    filtered = filter_transients(summed, 3)
    smoothed = np.convolve(filtered, np.ones(6))
    smoothed /= np.max(smoothed)
    peaks, info = scipy.signal.find_peaks(smoothed, height=0.2, prominence=0.15, distance=10)
    amp = info["peak_heights"]
    if verbose:
        plt.plot(smoothed)
        plt.scatter(peaks, amp, c='red')
        plt.show()
    times = peaks.astype(float)
    times -= times[0]
    # Interestingly, normalizing times seems to make the results slightly worse.
    # (But this may be simply becuase stroke identification is not very good yet.)
    if len(times) > 1:
        times /= times[-1]
    # Matrix of [[stroke time, stroke amplitude]]
    return np.vstack((times, amp)).T

def preprocessor(letters, fs):
    strokes = np.array([find_strokes(letter) for letter in letters], dtype=object)
    # Since letters have different numbers of strokes, pad to get consistent feature vector length.
    max_strokes = len(max(strokes, key=len))
    stroke_matrix = np.empty((len(strokes), max_strokes, 2))
    for i, stroke in enumerate(strokes):
        stroke_matrix[i] = np.vstack((stroke, np.ones((max_strokes - len(stroke), 2)) * 1000))
    return stroke_matrix.reshape((-1, max_strokes * 2))


if __name__ == '__main__':
    eval.run(preprocessor, KNeighborsClassifier(1))
