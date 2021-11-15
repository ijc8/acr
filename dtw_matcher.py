import numpy as np
from dtw import dtw

def block_audio(x, blockSize, hopSize):
    numBlocks = (x.size - blockSize) // hopSize
    xb = np.empty((numBlocks, blockSize))
    for i in range(numBlocks):
        xb[i] = x[i*hopSize:i*hopSize + blockSize]
    return xb

def get_power(x):
    # return np.log10((block_audio(np.diff(x), 2048, 512)**2).sum(axis=1).clip(1e-10, 1)) * 10
    xb = block_audio(np.diff(x), 2048, 512)
    # xb *= np.hanning(xb.shape[1])[None, :]
    return np.log10((xb**2).sum(axis=1).clip(1e-10, 1)) * 10
    # spec = np.abs(np.fft.rfft(xb))[:, :22]
    # return (spec**2).sum(axis=1)
    # return np.log10((spec**2).sum(axis=1).clip(1e-10, 1)) * 10

def preprocessor(letters, fs):
    global power
    power = np.empty(letters.shape, dtype=object)
    for i, letter in enumerate(letters):
        power[i] = get_power(letter)
        # power[i] = (power[i] - power[i].mean()) / power[i].std()
    return np.arange(len(power)).reshape(-1, 1)

from dtw import rabinerJuangStepPattern, asymmetric

def dtw_dist(a, b):
    a = power[int(a[0])]
    b = power[int(b[0])]
    return dtw(a, b, dist_method="euclidean", distance_only=True).normalizedDistance

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

spec, *_ = plt.specgram(np.diff(letters[0,0,0]), NFFT=512, noverlap=0);
plt.yticks(np.arange(0, 250, 10))
plt.imshow(np.log(spec), origin='lower')
plt.plot(np.log10(spec.sum(axis=0)))
# Pretty good way to find non-stroke sounds:
cutoff = 22
plt.plot(np.min(spec[cutoff:150], axis=0) / np.max(spec[:cutoff], axis=0))

def match_sequence(X, y, signal):
    position = 0
    seq = []
    while position < len(signal):
        print(f"position is {position} / {len(signal)}")
        scores = np.empty(len(X))
        indices = np.empty(len(X), dtype=int)
        for i in range(len(X)):
            template = power[X[i, 0]]
            alignment = dtw(template, signal[position:], dist_method="euclidean", open_end=True)
            scores[i] = alignment.normalizedDistance
            indices[i] = alignment.jmin
        best = scores.argmin()
        length = indices[best] + 1
        print("best is", best, "label is", y[best], "length is", length)
        seq.append((y[best], position, position + length))
        position += length
    return seq

def match_sequence2(X, y, signal):
    templates = power[X[:, 0]]
    template_lengths = np.array([template.size for template in templates])
    template_starts = np.concatenate(([0], np.cumsum(template_lengths[:-1])))
    template_concat = np.concatenate(templates)
    distances = np.empty((signal.size, template_concat.size))
    costs = np.empty((signal.size + 1, template_concat.size))
    costs[0] = np.inf
    print(template_starts, costs.shape)
    costs[0, template_starts] = 0
    pointers = np.empty((signal.size, template_concat.size, 2))
    for i in range(1, signal.size + 1):
        distances[i-1] = np.abs(template_concat - signal[i-1])
        for start, length in zip(template_starts, template_lengths):
            for j in range(length):
                possibilities = []
                if i > 0:
                    possibilities.append((i-1, start+j))
                    if j == 0:
                        # Could come from the end of any other template.
                        for s, l in zip(template_starts, template_lengths):
                            possibilities.append((i-1, s+l-1))
                if j > 0:
                    possibilities.append((i, start+j-1))
                if i > 0 and j > 0:
                    possibilities.append((i-1, start+j-1))
                r, c = min(possibilities, key=lambda p: costs[p[0], p[1]])
                costs[i, start+j] = distances[i-1, start+j] + costs[r, c]
                pointers[i-1, start+j] = [r, c]
    plt.imshow(distances.T, origin='lower', aspect='auto')
    plt.show()
    costs = costs[1:]
    plt.imshow(costs.T, origin='lower', aspect='auto')
    return costs

word = np.concatenate((a_power, b_power))

single_templates = X[indices[:, 2] == 0]
costs = match_sequence2(single_templates, None, word)

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
extra[0] = np.array([-35])
power = np.concatenate((power, extra))
X_train = np.concatenate((X_train, [[-1]]))
y_train = np.concatenate((y_train, [-1]))

indices[[X_test[y_test == 0][0, 0]]]
a_test = letters.reshape(-1)[X_test[y_test == 0][0, 0]]
b_test = letters.reshape(-1)[X_test[y_test == 1][0, 0]]
plt.specgram(a_test);
plt.plot(get_power(a_test));
plt.plot(get_power(b_test));
a_power = get_power(a_test)
b_power = get_power(b_test)
word = np.concatenate((a_power, a_power[-1] * np.ones(100), b_power))

X_train[25]
plt.plot(word)
alignment = dtw(power[-1], get_power(word)[138:], open_end=True, keep_internals=True)
alignment.plot('threeway')
alignment.normalizedDistance

match_sequence(X_train, y_train, word)

match_sequence(X_train, y_train, a_power)

from dtw import dtw
import dtw
dtw.asymmetric.plot()
dtw.rabinerJuangStepPattern(1, "c").plot()
print(dtw.asymmetric)

alignment = dtw.dtw(power[0], word, step_pattern="symmetric2", open_begin=True, open_end=True, keep_internals=True)
alignment.plot("twoway", offset=25)
plt.imshow(alignment.costMatrix, origin='lower')


scores.argmin()
y[0]
positions[0]
len(power[0])
template = power[0]
alignment = dtw(template, word_power, dist_method="euclidean", open_end=True)
alignment.normalizedDistance
alignment.plot()
