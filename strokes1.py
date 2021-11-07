import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
import scipy.signal

from sklearn.neighbors import RadiusNeighborsClassifier

import evaluate


def filter_transients(x, n):
    return np.median(x[:len(x)//n*n].reshape((-1, n)), axis=1)

def find_strokes(x, verbose=0):
    X = mlab.specgram(np.diff(x), NFFT=512, noverlap=256)[0]
    summed = np.sum(X[:120], axis=0)
    filtered = summed # filter_transients(summed, 2)
    smoothed = np.convolve(filtered, np.ones(2))
    smoothed /= np.max(smoothed)
    peaks, info = scipy.signal.find_peaks(smoothed, height=0.2, prominence=0.15, distance=10)
    amp = info["peak_heights"]
    if verbose:
        if verbose > 1:
            plt.plot(np.arange(len(summed)) / 2, summed / np.max(summed))
            plt.plot(filtered / np.max(filtered))
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

def just_stroke_count(letters, fs):
    strokes = np.array([find_strokes(letter) for letter in letters], dtype=object)
    return np.array([len(stroke) for stroke in strokes]).reshape((-1, 1))

def just_stroke_times(letters, fs):
    strokes = np.array([find_strokes(letter) for letter in letters], dtype=object)
    # Since letters have different numbers of strokes, pad to get consistent feature vector length.
    max_strokes = len(max(strokes, key=len))
    stroke_matrix = np.empty((len(strokes), max_strokes))
    for i, stroke in enumerate(strokes):
        stroke_matrix[i] = np.hstack((stroke[:, 0], np.ones(max_strokes - len(stroke)) * 1000))
    return stroke_matrix

def just_stroke_amps(letters, fs):
    strokes = np.array([find_strokes(letter) for letter in letters], dtype=object)
    # Since letters have different numbers of strokes, pad to get consistent feature vector length.
    max_strokes = len(max(strokes, key=len))
    stroke_matrix = np.empty((len(strokes), max_strokes))
    for i, stroke in enumerate(strokes):
        stroke_matrix[i] = np.hstack((stroke[:, 1], np.ones(max_strokes - len(stroke)) * 1000))
    return stroke_matrix

def stroke_times_and_amps(letters, fs):
    strokes = np.array([find_strokes(letter) for letter in letters], dtype=object)
    # Since letters have different numbers of strokes, pad to get consistent feature vector length.
    max_strokes = len(max(strokes, key=len))
    features = strokes[0].shape[1]
    stroke_matrix = np.empty((len(strokes), max_strokes, features))
    for i, stroke in enumerate(strokes):
        stroke_matrix[i] = np.vstack((stroke, np.ones((max_strokes - len(stroke), features)) * 1000))
    return stroke_matrix.reshape((-1, max_strokes * features))

if __name__ == '__main__':
    # RadiusNeighbors seems to do a bit better than KNeighbors in general,
    # and especially for stroke count (which is 1D).
    print("== just stroke count ==")
    evaluate.run(just_stroke_count, RadiusNeighborsClassifier(0))
    print("== just stroke times ==")
    evaluate.run(just_stroke_times, RadiusNeighborsClassifier(1))
    print("== just stroke amps ==")
    evaluate.run(just_stroke_amps, RadiusNeighborsClassifier(1))
    print("== stroke times & amps ==")
    evaluate.run(stroke_times_and_amps, RadiusNeighborsClassifier(1))



letters, fs = evaluate.load_dataset()

X = stroke_times_and_amps(letters.reshape(-1), fs)
y = np.indices(letters.shape)[1].reshape(-1)
subjects = np.indices(letters.shape)[0].reshape(-1)
indices = np.indices(letters.shape).reshape((letters.ndim, -1)).T

classifier = RadiusNeighborsClassifier(1)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(2)

mask = subjects == 0
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(
    X[mask], y[mask], indices[mask],
    test_size=0.25,
    stratify=y[mask]
)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"{round((y_pred == y_test).mean() * 100, 2)}%")
for i in range(len(evaluate.alphabet)):
    mask = y_test == i
    print(f"- {evaluate.alphabet[i]}: {round((y_pred[mask] == y_test[mask]).mean() * 100, 2)}%")
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=evaluate.alphabet,
    include_values=True
)

idx = np.where(y_test == evaluate.alphabet.index('O'))[0]
idx
X_test[45]
evaluate.alphabet[23]
classifier.predict(X_test[idx])
index_test[idx]
X_test[idx].astype(int)

classifier.predict([X_test[108]])

plt.specgram(letters[0, 14, 15], NFFT=512, noverlap=256);
find_strokes(letters[0, 14, 15], verbose=True)

idx = np.where(y_train == evaluate.alphabet.index('O'))[0]
index_train[idx]
find_strokes(letters[0, 14, 5], verbose=True)
X_train[idx].astype(int)

idx = np.where(y_train == evaluate.alphabet.index('A'))[0]
index_train[idx]

f, (a, b) = plt.subplots(2)
X = mlab.specgram(np.diff(letters[0, 0, 0]), NFFT=512, noverlap=511)[0]
a.imshow(np.log(X), origin='lower', aspect='auto')
b.plot(X[:120].sum(axis=0))
f.show()

find_strokes(letters[0, 0, 0], verbose=2)
X_train[idx].astype(int)

proba = classifier.predict_proba([[0, 1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]])[0]
list(zip(evaluate.alphabet, proba))
