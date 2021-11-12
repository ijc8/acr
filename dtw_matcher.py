import math
import numpy as np
from dtw import dtw

def block_audio(x, blockSize, hopSize):
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb

def preprocessor(letters, fs):
    global power
    power = np.empty(letters.shape, dtype=object)
    for i, letter in enumerate(letters):
        letter = letter.astype(float) / np.iinfo(np.int32).max
        power[i] = np.log((block_audio(np.diff(letter), 2048, 512)**2).sum(axis=1))
    return np.arange(len(power)).reshape(-1, 1)

def dtw_dist(a, b):
    a = power[int(a[0])]
    b = power[int(b[0])]
    return dtw(a, b, dist_method="euclidean").normalizedDistance

from sklearn.neighbors import KNeighborsClassifier
cls = KNeighborsClassifier(1, metric=dtw_dist, algorithm='brute', n_jobs=8)

import evaluate
evaluate.run(preprocessor, cls, subset=np.arange(4))

# Single-subject accuracy (0): 95.0%
# Single-subject accuracy (1): 90.0%
# Single-subject accuracy (2): 100.0%
# Single-subject accuracy (3): 100.0%
# All-subject accuracy: 97.5%
# - Subject 0: 100.0%
# - Subject 1: 94.74%
# - Subject 2: 95.24%
# - Subject 3: 100.0%
# Left-out-subject accuracy (0): 82.5%
# Left-out-subject accuracy (1): 66.25%
# Left-out-subject accuracy (2): 78.75%
# Left-out-subject accuracy (3): 70.0%
