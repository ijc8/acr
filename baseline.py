import numpy as np
import scipy.signal

from sklearn.neighbors import KNeighborsClassifier

import evaluate


processed = None

def preprocessor(letters, fs):
    global processed
    if processed is None:
        def process(x):
            filtered = np.convolve(np.convolve(np.abs(np.diff(x)), np.ones(4096)), np.ones(1024))
            # What about dividing by peak rather than std? (Get range from 0-1, or maybe -1-1.)
            return (filtered - np.mean(filtered)) / np.std(filtered)
        processed = np.array([process(letter) for letter in letters], dtype=object)
    return np.arange(len(processed)).reshape((-1, 1))

def distance(a, b):
    a, b = processed[int(a[0])], processed[int(b[0])]
    l = max(len(a), len(b))
    corr = np.max(scipy.signal.correlate(a, b, mode='full')) / l
    return -corr


if __name__ == '__main__':
    evaluate.run(preprocessor, KNeighborsClassifier(1, metric=distance, algorithm='brute', n_jobs=8))
