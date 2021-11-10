import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import evaluate


def extract_spans(x, verbose=0, db=-11, smooth=19, cutoff=22):
    X = mlab.specgram(np.diff(x), NFFT=512, noverlap=256)[0]
    X -= np.median(X, axis=1)[:, None]
    # Filter spectrum when finding spans.
    # (Useful for avoiding paper noises.)
    summed = signal.medfilt(X[:cutoff].sum(axis=0), smooth)
    summed /= np.max(summed)

    # Use full spectrum when collecting amplitude data.
    full = signal.medfilt(X.sum(axis=0), smooth)
    full /= np.max(full)

    threshold = 10**(db/10)
    active = (summed > threshold) & (full > threshold)
    indices = np.where(np.diff(np.concatenate(([False], active, [False]))))[0]
    if verbose:
        print(indices)
    spans = np.empty((len(indices) - 1, 3))
    for i, (start, end) in enumerate(zip(indices, indices[1:])):
        # Time, duration, amplitude.
        spans[i] = [start, end - start, full[start:end].mean()]

    if verbose:
        if verbose > 1:
            plt.plot(np.log10(summed))
            plt.show()
        plt.plot(summed)
        plt.plot(full)
        plt.axhline(threshold)
        print(indices)
        for start, duration, amp in spans:
            color = 'red' if amp >= threshold else 'blue'
            plt.axvline(start, color=color)
            plt.axvspan(start, start + duration, color=color, alpha=0.1)
        plt.scatter(spans[:, 0] + spans[:, 1] / 2, spans[:, 2])
        plt.show() 

    spans[:, 0] -= spans[0, 0]
    total_duration = spans[-1, 0] + spans[-1, 1]
    spans[:, 0:2] /= total_duration

    # Matrix of [[time, duration, amplitude]]
    return spans

def just_span_count(letters, fs):
    spans = np.array([extract_spans(letter) for letter in letters], dtype=object)
    return np.array([len(span) for span in spans]).reshape((-1, 1))

def span_features(features):
    def get_features(letters, fs):
        spans = np.array([extract_spans(letter) for letter in letters], dtype=object)
        # Since letters have different numbers of spans, pad to get consistent feature vector length.
        max_spans = len(max(spans, key=len))
        span_matrix = np.empty((len(spans), max_spans, len(features)))
        for i, span in enumerate(spans):
            span_matrix[i] = np.vstack((span[:, features], np.ones((max_spans - len(span), len(features))) * 1000))
        return span_matrix.reshape((-1, max_spans * len(features)))
    return get_features

def alt_dist_metric(a, b, verbose=0):
    # TODO: Consider sum-of-differences (current approach) vs. normalized dot-product.
    spanA, spanB = spans[int(a[0])], spans[int(b[0])]
    i = j = 0
    score = 0
    if verbose:
        for start, dur, amp in spanA:
            plt.axvspan(start, start + dur, 0, amp, color='red', alpha=0.5)
        for start, dur, amp in spanB:
            plt.axvspan(start, start + dur, 0, amp, color='blue', alpha=0.5)
    while i < len(spanA) and j < len(spanB):
        startA, endA, ampA = spanA[i, 0], spanA[i, 0] + spanA[i, 1], spanA[i, 2]
        startB, endB, ampB = spanB[j, 0], spanB[j, 0] + spanB[j, 1], spanB[j, 2]
        if startA <= endB and endA >= startB:
            overlap = min(endA, endB) - max(startA, startB)
            score += abs(overlap * (ampA - ampB))
        if endA <= endB:
            i += 1
        else:
            j += 1
    return score

def span_special(letters, fs):
    global spans
    spans = np.array([extract_spans(letter) for letter in letters.reshape(-1)], dtype=object)
    return np.arange(len(spans)).reshape((-1, 1))

if __name__ == '__main__':
    # RadiusNeighbors seems to do a bit better than KNeighbors in general,
    # and especially for span count (which is 1D).
    print("== just span count ==")
    evaluate.run(just_span_count, RadiusNeighborsClassifier(0))
    print("== just span times ==")
    evaluate.run(span_features([0]), KNeighborsClassifier(9))
    print("== just span durations ==")
    evaluate.run(span_features([1]), KNeighborsClassifier(9))
    print("== just span amps ==")
    evaluate.run(span_features([2]), KNeighborsClassifier(9))
    print("== span times & amps ==")
    evaluate.run(span_features([0, 2]), KNeighborsClassifier(9))
    print("== span times & durations & amps ==")
    evaluate.run(span_features([0, 1, 2]), KNeighborsClassifier(9))
    print("== and finally, with a fancier classifier ==")
    evaluate.run(span_features([0, 1, 2]), RandomForestClassifier())
    print("== custom distance metric ==")
    evaluate.run(span_special, KNeighborsClassifier(1, algorithm="brute", metric=alt_dist_metric))
