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

def get_power(x):
    return np.log10((block_audio(np.diff(x), 2048, 512)**2).sum(axis=1).clip(1e-10, 1)) * 10

def preprocessor(letters, fs):
    global power
    power = np.empty(letters.shape, dtype=object)
    for i, letter in enumerate(letters):
        power[i] = get_power(letter)
    return np.arange(len(power)).reshape(-1, 1)

def dtw_dist(a, b):
    a = power[int(a[0])]
    b = power[int(b[0])]
    return dtw(a, b, dist_method="euclidean").normalizedDistance

from sklearn.neighbors import KNeighborsClassifier
cls = KNeighborsClassifier(1, metric=dtw_dist)

import evaluate
evaluate.run(preprocessor, cls, np.arange(4))

# With four classes (ABCD):
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

# With eight classes (ABCDEFGH):
# Single-subject accuracy (0): 90.0%
# Single-subject accuracy (1): 82.5%
# Single-subject accuracy (2): 90.0%
# Single-subject accuracy (3): 90.0%
# All-subject accuracy: 88.75%
# - Subject 0: 93.18%
# - Subject 1: 87.18%
# - Subject 2: 90.48%
# - Subject 3: 82.86%
# Left-out-subject accuracy (0): 50.0%
# Left-out-subject accuracy (1): 44.38%
# Left-out-subject accuracy (2): 40.0%
# Left-out-subject accuracy (3): 39.38%

# With the whole shebang (A-Z):
# Single-subject accuracy (0): 75.38%
# Single-subject accuracy (1): 68.46%
# Single-subject accuracy (2): 74.62%
# Single-subject accuracy (3): 66.15%
# All-subject accuracy: 64.42%
# - Subject 0: 72.46%
# - Subject 1: 64.62%
# - Subject 2: 66.41%
# - Subject 3: 52.89%
# Left-out-subject accuracy (0): 19.42%
# Left-out-subject accuracy (1): 19.42%
# Left-out-subject accuracy (2): 14.81%
# Left-out-subject accuracy (3): 12.69%


def parse_word(X, y, word):
    word_power = get_power(word)
    position = 0
    seq = []
    while position < len(word_power):
        print(f"position is {position} / {len(word_power)}")
        scores = np.empty(len(X))
        indices = np.empty(len(X), dtype=int)
        for i in range(len(X)):
            template = power[X[i, 0]]
            # print(template.shape, word_power[position:].shape)
            alignment = dtw(template, word_power[position:], dist_method="euclidean", open_end=True)
            scores[i] = alignment.normalizedDistance
            indices[i] = alignment.jmin
        best = scores.argmin()
        length = indices[best] + 1
        print("best is", best, "label is", y[best], "length is", length)
        seq.append((y[best], position, position + length))
        position += length
    return seq

word = np.concatenate((letters[0,0,0], letters[0,1,0]))

letters, fs = evaluate.load_dataset()
letters = letters[:1, :4, :]
X = preprocessor(letters.reshape(-1), fs)
y = np.indices(letters.shape)[1].reshape(-1)
subjects = np.indices(letters.shape)[0].reshape(-1)
indices = np.indices(letters.shape).reshape((letters.ndim, -1)).T

mask = np.ones(len(X), dtype=bool)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X[mask], y[mask], indices[mask], test_size=0.25, stratify=y[mask],
)

# Add a spacing template to match silence.
extra = np.empty((1,), dtype=object)
extra[0] = np.array([-34])
power[-1][0] = -100
power = np.concatenate((power, extra))
X_train = np.concatenate((X_train, [[-1]]))
y_train = np.concatenate((y_train, [-1]))

def tape(pieces, env_size=512):
    # Concatenate audio signals safely (without discontinuities at the boundaries).
    enveloped = []
    envelope = np.linspace(0, 1, env_size, endpoint=False)
    for piece in pieces:
        piece = piece.copy()
        piece[:env_size] *= envelope
        piece[-env_size:] *= envelope[::-1]
        enveloped.append(piece)
    return np.concatenate(enveloped)

indices[[X_test[y_test == 0][0, 0]]]
a_test = letters.reshape(-1)[X_test[y_test == 0][0, 0]]
b_test = letters.reshape(-1)[X_test[y_test == 1][0, 0]]
plt.specgram(a_test);
plt.plot(get_power(a_test));
plt.plot(get_power(b_test));
word = tape((a_test, np.zeros(44100), b_test))
plt.plot(get_power(word)[138:354]);

X_train[25]
plt.plot(power[77])
alignment = dtw(power[-1], get_power(word)[138:], open_end=True, keep_internals=True)
alignment.plot('threeway')
alignment.normalizedDistance

parse_word(X_train, y_train, word)

parse_word(X_train, y_train, a_test)

scores.argmin()
y[0]
positions[0]
len(power[0])
template = power[0]
alignment = dtw(template, word_power, dist_method="euclidean", open_end=True)
alignment.normalizedDistance
alignment.plot()
