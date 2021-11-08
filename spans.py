import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import evaluate


a = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0])
mask = a > 0.3
indices = np.where(np.diff(np.concatenate(([False], mask, [False]))))[0]
indices
np.split(a, indices)

indices = np.array([3,7,8,11])
indices
mask = np.diff(indices) > 2
np.concatenate(([True], mask)) & np.concatenate((mask, [True]))

def extract_spans(x, verbose=0, db=-7, smooth=11, cutoff=25):
    X = mlab.specgram(np.diff(x), NFFT=512, noverlap=256)[0]
    X -= np.median(X, axis=1)[:, None]
    summed = signal.medfilt(X[:cutoff].sum(axis=0), smooth)
    summed /= np.max(summed)

    threshold = 10**(db/10)
    active = summed > threshold
    indices = np.where(np.diff(np.concatenate(([False], active, [False]))))[0]
    if verbose:
        print(indices)
    spans = np.empty((len(indices) - 1, 3))
    for i, (start, end) in enumerate(zip(indices, indices[1:])):
        # Time, duration, amplitude.
        spans[i] = [start, end - start, summed[start:end].mean()]

    if verbose:
        if verbose > 1:
            plt.plot(np.log10(summed))
            plt.show()
        plt.plot(summed)
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


# Misc. experimentation, investigation, tweaking
letters, fs = evaluate.load_dataset()
spans = np.array([extract_spans(letter) for letter in letters.reshape(-1)], dtype=object)
classifier = KNeighborsClassifier(1, algorithm="brute", metric=alt_dist_metric)
X = np.arange(len(spans)).reshape((-1, 1))

X = just_span_count(letters.reshape(-1), fs)
max(X)
indices[np.argmax(X)]
indices

y = np.indices(letters.shape)[1].reshape(-1)
subjects = np.indices(letters.shape)[0].reshape(-1)
indices = np.indices(letters.shape).reshape((letters.ndim, -1)).T

extract_spans(letters[0,0,0], verbose=1)
extract_spans(letters[0,0,1], verbose=1, smooth=11, db=-7)

alt_dist_metric([0], [1], verbose=True)

classifier = RadiusNeighborsClassifier(0)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
mask = subjects == 0
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(
    X[mask], y[mask], indices[mask],
    test_size=0.25,
    stratify=y[mask]
)

classifier = KNeighborsClassifier(9)

index_train[y_train == evaluate.alphabet.index('M')]
index_test[y_test == evaluate.alphabet.index('M')]
indices[240]
indices[247]
extract_spans(letters[0, 12, 0], verbose=1, cutoff=25, db=-8)
extract_spans(letters[0, 12, 10], verbose=1, cutoff=25, db=-5)
alt_dist_metric([240], [250], verbose=True)

indices[182]
indices[188]
extract_spans(letters[0, 9, 2], verbose=1, cutoff=25, db=-5)
extract_spans(letters[0, 9, 8], verbose=1, cutoff=25, db=-5)
alt_dist_metric([182], [188], verbose=True)


index_train[y_train == evaluate.alphabet.index('N')]
index_test[y_test == evaluate.alphabet.index('N')]
indices[260]
indices[262]
extract_spans(letters[0, 13, 0], verbose=1, cutoff=25, db=-8)
extract_spans(letters[0, 13, 2], verbose=1, cutoff=25, db=-8)
alt_dist_metric([260], [262], verbose=True)



classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"{round((y_pred == y_test).mean() * 100, 2)}%")
for i in range(len(evaluate.alphabet)):
    mask = y_test == i
    print(f"- {evaluate.alphabet[i]}: {round((y_pred[mask] == y_test[mask]).mean() * 100, 2)}%")
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=evaluate.alphabet,
    include_values=False
)

classifier.predict(X_test[[55, 73]])
index_test[55]

idx = np.where(y_test == evaluate.alphabet.index('J'))[0]
idx
X_test[45]
evaluate.alphabet[22]
classifier.predict(X_test[idx])
index_test[idx]
X_test[idx].astype(int)

classifier.predict([X_test[108]])
from IPython.display import Audio

display(Audio(letters[0, 9, 3], rate=fs))
plt.specgram(letters[0, 9, 3], NFFT=512, noverlap=256);
plt.xlim(50, 120)
extract_spans(letters[0, 9, 3], 2, dist=11, db=-9)

idx = np.where(y_train == evaluate.alphabet.index('O'))[0]
index_train[idx]
extract_spans(letters[0, 14, 5], verbose=1)
X_train[idx].astype(int)

plt.specgram(letters[1, 8, 14], NFFT=512, noverlap=256);
extract_spans(letters[1, 10, 1], verbose=1)

idx = np.where(y_train == evaluate.alphabet.index('A'))[0]
index_train[idx]

f, (a, b, c) = plt.subplots(3, figsize=(20, 20), sharex=True)
X = mlab.specgram(np.diff(letters[0, 12, 10]), NFFT=512, noverlap=256)[0]
X -= np.median(X, axis=1)[:, None]
a.imshow(np.log(X), origin='lower', aspect='auto')
summed = X[:25].sum(axis=0)
summed = signal.medfilt(summed, 7)
summed /= np.max(summed)
b.plot(summed)
b.axhline(10**(-7/10))
c.plot(np.log(summed))
f.show()



extract_spans(letters[1, 10, 1], verbose=1, cutoff=120)
X_train[idx].astype(int)

proba = classifier.predict_proba([[0, 1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]])[0]
list(zip(evaluate.alphabet, proba))
