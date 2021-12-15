import numpy as np
from scipy import stats
import scipy.io.wavfile
import matplotlib.pyplot as plt
import os
from numba import jit


def block_audio(x, blockSize, hopSize):
    numBlocks = (x.size - blockSize) // hopSize
    xb = np.empty((numBlocks, blockSize))
    for i in range(numBlocks):
        xb[i] = x[i*hopSize:i*hopSize + blockSize]
    return xb

def get_power(x):
    xb = block_audio(np.diff(x), 2048, 512)
    return np.log10(np.maximum((xb**2).mean(axis=1), 1e-10)) * 10

def get_spec(x):
    # return np.log10(mlab.specgram(np.diff(x), 512, noverlap=0)[0].T)
    return np.abs(np.fft.rfft(block_audio(np.diff(x), 2048, 512) * np.hanning(2048)[None, :], axis=1)[:, 0:1])

def get_spec_diff(x):
    # return np.log10(mlab.specgram(np.diff(x), 512, noverlap=0)[0].T)
    return np.diff(np.abs(np.fft.rfft(block_audio(np.diff(x), 512, 512), axis=1)))

def preprocessor(letters):
    global power
    power = np.empty(letters.shape, dtype=object)
    for i, letter in enumerate(letters):
        power[i] = get_power(letter)
    return np.arange(len(power)).reshape(-1, 1)


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

def match_sequence(templates, labels, signal, plot=False, zscore=True, metric='euclidean'):
    templates = templates[:]
    if zscore:
        for i in range(len(templates)):
            templates[i] = stats.zscore(templates[i])
        signal = stats.zscore(signal)

    template_lengths = np.array([len(template) for template in templates])
    template_starts = np.concatenate(([0], np.cumsum(template_lengths[:-1])))
    template_ends = np.cumsum(template_lengths) - 1
    template_concat = np.concatenate(templates)
    if signal.ndim < 2:
        signal = signal[:, None]
    if template_concat.ndim < 2:
        template_concat = template_concat[:, None]
    distances = scipy.spatial.distance.cdist(signal, template_concat, metric=metric)

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
    predicted = [labels[i] for i in template_seq]
    if plot:
        plt.show()
    return predicted

if False:
    files = os.listdir("good")
    X = []
    y = []

    for file in files:
        fs, x = scipy.io.wavfile.read(os.path.join("good", file))
        x = x.astype(float) / np.iinfo(np.int32).max
        # X.append(get_power(x)[:, None])
        X.append(get_spec(x))
        y.append(os.path.splitext(file)[0])

    fs, word = scipy.io.wavfile.read("words/the.wav")
    word = word[:512*275]
    word = word.astype(float) / np.iinfo(np.int32).max

    x = word
    xb = block_audio(np.diff(x), 2048, 512)
    xb.shape

    plt.plot(np.sqrt((xb**2).mean(axis=1)))
    plt.imshow(np.abs(np.fft.rfft(xb, axis=1).T), origin='lower', aspect='auto')
    plt.plot(np.fft.rfft(xb, axis=1)[:, 0])

    window = np.hanning(2048)[None, :]
    F = np.abs(np.fft.rfft(xb * window, axis=1))
    F.shape
    plt.plot(np.log10(F[:, 0]))
    plt.imshow(np.log10(F.T), origin='lower', aspect='auto')

    # word = get_power(word)[:, None]
    word = get_spec(word)
    match_sequence(X, y, word, zscore=False, plot=True, metric='euclidean')
    # T, H, T, T - uh oh.
    y
    plt.plot(get_spec(word))
    plt.plot(np.concatenate((X[1], X[0], X[1], X[1])))
    plt.plot(np.concatenate((X[1], X[0], X[2])))
    plt.imshow(np.log10(word.T), origin='lower', aspect='auto')
    plt.imshow(X[1].T, origin='lower', aspect='auto')
