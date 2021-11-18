import numpy as np
from dtw import dtw
from scipy import stats
import evaluate

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

def dtw_dist(a, b):
    a = power[int(a[0])]
    b = power[int(b[0])]
    return dtw(a, b, dist_method="euclidean", distance_only=True).normalizedDistance

if False:
    from sklearn.neighbors import KNeighborsClassifier
    cls = KNeighborsClassifier(1, metric=dtw_dist)
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

if False:
    spec, *_ = plt.specgram(np.diff(letters[0,0,0]), NFFT=512, noverlap=0);
    plt.yticks(np.arange(0, 250, 10))
    plt.imshow(np.log(spec), origin='lower')
    plt.plot(np.log10(spec.sum(axis=0)))
    # Pretty good way to find non-stroke sounds:
    cutoff = 22
    plt.plot(np.min(spec[cutoff:150], axis=0) / np.max(spec[:cutoff], axis=0))

def match_sequence_naive(X, y, signal):
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

from numba import jit



@jit(nopython=True)
def compute_cost_matrix(distances, template_starts, template_lengths, template_ends):
    costs = np.empty_like(distances)
    pointers = np.empty_like(distances, dtype=np.int_)
    template_transitions = np.empty((len(template_ends) + 1), dtype=np.int_)
    template_transitions[1:] = template_ends

    # Steps:
    # 0: Down -> Up (i, j-1)
    # 1: Left -> Right (i-1, j)
    # 2: Diagonal (i-1, j-1)
    # Transitions between templates are represented by 3 + the previous template's index.

    for start, length in zip(template_starts, template_lengths):
        # Only one step is possible: must come from below.
        costs[0, start:start+length] = np.cumsum(distances[0, start:start+length])
        pointers[0, start+1:start+length] = 0
    for i in range(1, distances.shape[0]):
        for start, length in zip(template_starts, template_lengths):
            # Could come from the left, or from the end of any other template.
            template_transitions[0] = start
            step_costs = costs[i-1][template_transitions]
            best = np.argmin(step_costs)
            costs[i, start] = distances[i, start] + step_costs[best]
            pointers[i, start] = 1 if best == 0 else best + 2

            for j in range(start + 1, start + length):
                # Could come from any direction (down, left, diagonal).
                dist = distances[i, j]
                step_costs = np.array([
                    dist + costs[i, j-1],
                    dist + costs[i-1, j],
                    2*dist + costs[i-1, j-1],
                ])
                best = np.argmin(step_costs)
                costs[i, j] = step_costs[best]
                pointers[i, j] = best
    return costs, pointers

def backtrack(costs, pointers, template_starts, template_ends, plot=False):
    # Backtrack to find best path through cumulative cost matrix.
    # (This just returns the sequence of templates, not every single index in the path.)
    pos = np.array([costs.shape[0] - 1, template_ends[costs[-1, template_ends].argmin()]])
    steps = np.array([
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    transitions = []
    while not (pos[0] == 0 and pos[1] in template_starts):
        step = pointers[pos[0], pos[1]]
        if step < len(steps):
            parent = pos - steps[step]
        else:
            parent = np.array([pos[0] - 1, template_ends[step - 3]])
            transitions.append(pos)
        if plot:
            delta = parent - pos
            plt.arrow(
                pos[0], pos[1], delta[0], delta[1],
                color='red', head_width=0.05, length_includes_head=True,
            )          
        pos = parent
    transitions.append(pos)
    return transitions

def match_sequence(X, y, signal, plot=False, zscore=True):
    templates = power[X[:, 0]]
    if zscore:
        # Experimenting with z-normalization here.
        for i in range(templates.size):
            templates[i] = stats.zscore(templates[i])
        signal = stats.zscore(signal)
    template_lengths = np.array([template.size for template in templates])
    template_starts = np.concatenate(([0], np.cumsum(template_lengths[:-1])))
    template_ends = np.cumsum(template_lengths) - 1
    template_concat = np.concatenate(templates)
    distances = np.abs(template_concat - signal[:, None])

    costs, pointers = compute_cost_matrix(distances, template_starts, template_lengths, template_ends)
    if plot:
        plt.figure()
        plt.imshow(distances.T, origin='lower', aspect='auto', interpolation='none')
        plt.show()
        plt.figure()
        plt.imshow(costs.T, origin='lower', aspect='auto')
        for s in template_starts:
            plt.axhline(s, color='orange', alpha=0.3)

    transitions = backtrack(costs, pointers, template_starts, template_ends, plot)
    template_seq = [template_starts.tolist().index(p[1]) for p in transitions[::-1]]
    predicted = y[template_seq]
    if plot:
        plt.show()
    return predicted


import random
import time

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

def evaluate_sequence(length, **kwargs):
    "Generate a random 'word' from testing letters and try to decode it using training letters."
    indices = np.array([random.randrange(len(X_test)) for _ in range(length)])
    word = np.concatenate([power[X_test[i, 0]] for i in indices])
    y_seq = y_test[indices]
    start = time.time()
    y_pred = match_sequence(X_train, y_train, word, **kwargs)
    end = time.time()
    word_time = ((len(word) - 1)*512 + 2048) / 44100
    calc_time = end - start
    print("word time:", word_time)
    print(f"calc time: {calc_time} ({round(calc_time / word_time * 100, 2)}% of realtime)")
    print("expected: ", y_seq)
    print("predicted:", y_pred)

if __name__ == '__main__':
    evaluate_sequence(5)
    evaluate_sequence(5)

if False:
    word = np.concatenate((a_power, b_power))

    plt.plot(np.concatenate((a_power, b_power)))
    plt.plot(np.concatenate((power[single_templates[0, 0]], power[single_templates[3, 0]])))
    plt.plot(np.concatenate((power[single_templates[0, 0]], power[single_templates[1, 0]])))

    # Here's the trouble.
    # This:
    plt.specgram(letters[0, 1, 16], NFFT=2048, noverlap=2048-512)
    plt.plot(get_power(letters[0, 1, 16]))
    # matches this:
    plt.specgram(letters[0, 3, 0], NFFT=2048, noverlap=2048-512)
    plt.plot(get_power(letters[0, 3, 0]))
    # when we'd like to match this:
    plt.specgram(letters[0, 1, 0], NFFT=2048, noverlap=2048-512)
    plt.plot(get_power(letters[0, 1, 0]))

    plt.plot(get_power(letters[0, 1, 16])[:112])
    plt.plot(get_power(letters[0, 1, 0])[:101])
    plt.plot(get_power(letters[0, 3, 0]))

    match_sequence(single_templates, single_labels, power[0], plot=True)
    dtw(get_power(letters[0, 1, 16])[:112], get_power(letters[0, 1, 0])[:101]).normalizedDistance
    dtw(get_power(letters[0, 1, 16])[:112], get_power(letters[0, 3, 0])).normalizedDistance
    dtw(get_power(letters[0, 1, 16])[:112], get_power(letters[0, 1, 0])[:101], keep_internals=True).plot("twoway")
    dtw(get_power(letters[0, 1, 16])[:112], get_power(letters[0, 3, 0]), keep_internals=True).plot("twoway")

    # Somehow, z-normalization fixes this.
    query = stats.zscore(get_power(letters[0, 1, 16])) # [:112])
    right = stats.zscore(get_power(letters[0, 1, 0])) # [:101])
    wrong = stats.zscore(get_power(letters[0, 3, 0]))
    plt.plot(query)
    plt.plot(right)
    plt.plot(wrong)
    dtw(query, right).normalizedDistance
    dtw(query, wrong).normalizedDistance

    # match_sequence(X_train, y_train, word)

    power[single_templates[1, 0]].shape
    power[single_templates[3, 0]].shape
    plt.plot(word)

    import matplotlib.pyplot as plt
    single_templates = X[indices[:, 2] == 0]
    single_labels = y[indices[:, 2] == 0]
    word = np.concatenate((power[single_templates[0, 0]], power[single_templates[3, 0]]))
    word = np.concatenate((a_power, b_power))
    costs = match_sequence2(single_templates, single_labels, word)
    costs = match_sequence2(X_train, y_train, word)

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
