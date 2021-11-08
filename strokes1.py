import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import evaluate


def filter_transients(x, n):
    return np.median(x[:len(x)//n*n].reshape((-1, n)), axis=1)

# NOTE: This has largely been superseded by `spans.py`.
# You would probably be better served by getting strokes by taking
# every other span from `extract_spans()`.
def find_strokes(x, verbose=0, db=-9, dist=11):
    X = mlab.specgram(np.diff(x), NFFT=512, noverlap=256)[0]
    summed = X[:120].sum(axis=0)
    summed /= np.max(summed)
    # summed = np.log10(summed) * 10

    active = summed > 10**(db/10)
    indices = np.where(active)[0]
    new_runs = np.where(np.diff(indices) > dist)[0]
    starts = indices[np.hstack(([0], new_runs + 1))]
    ends = indices[np.hstack((new_runs, [-1]))]

    times = starts.astype(float)  # t[peaks]
    times -= times[0]
    duration = (ends - starts).astype(float)
    if len(times) > 1:
        duration /= times[-1]
        times /= times[-1]
    amp = np.empty(len(times))
    for i, (start, end) in enumerate(zip(starts, ends)):
        amp[i] = summed[start:end+1].mean()

    if verbose:
        if verbose > 1:
            plt.plot(np.log10(summed))
            plt.show()
        plt.plot(summed)
        plt.plot(active)
        print(indices)
        print(np.diff(indices))
        print(starts, ends)
        for start, end in zip(starts, ends):
            plt.axvline(start, color='red')
            plt.axvspan(start, end, color='red', alpha=0.1)
        plt.scatter((starts + ends) / 2, amp)
        plt.show() 

    # Matrix of [[stroke time, stroke duration, stroke amplitude]]
    return np.vstack((times, duration, amp)).T

def just_stroke_count(letters, fs):
    strokes = np.array([find_strokes(letter) for letter in letters], dtype=object)
    return np.array([len(stroke) for stroke in strokes]).reshape((-1, 1))

def stroke_features(features):
    def get_features(letters, fs):
        strokes = np.array([find_strokes(letter) for letter in letters], dtype=object)
        # Since letters have different numbers of strokes, pad to get consistent feature vector length.
        max_strokes = len(max(strokes, key=len))
        stroke_matrix = np.empty((len(strokes), max_strokes, len(features)))
        for i, stroke in enumerate(strokes):
            stroke_matrix[i] = np.vstack((stroke[:, features], np.ones((max_strokes - len(stroke), len(features))) * 1000))
        return stroke_matrix.reshape((-1, max_strokes * len(features)))
    return get_features

if __name__ == '__main__':
    # RadiusNeighbors seems to do a bit better than KNeighbors in general,
    # and especially for stroke count (which is 1D).
    print("== just stroke count ==")
    evaluate.run(just_stroke_count, RadiusNeighborsClassifier(0))
    print("== just stroke times ==")
    evaluate.run(stroke_features([0]), KNeighborsClassifier(9))
    print("== just stroke durations ==")
    evaluate.run(stroke_features([1]), KNeighborsClassifier(9))
    print("== just stroke amps ==")
    evaluate.run(stroke_features([2]), KNeighborsClassifier(9))
    print("== stroke times & amps ==")
    evaluate.run(stroke_features([0, 2]), KNeighborsClassifier(9))
    print("== stroke times & durations & amps ==")
    evaluate.run(stroke_features([0, 1, 2]), KNeighborsClassifier(9))
    print("== and finally, with a fancier classifier ==")
    evaluate.run(stroke_features([0, 1, 2]), RandomForestClassifier())


# Misc. experimentation, investigation, tweaking
letters, fs = evaluate.load_dataset()

X = stroke_times_and_amps(letters.reshape(-1), fs)
y = np.indices(letters.shape)[1].reshape(-1)
subjects = np.indices(letters.shape)[0].reshape(-1)
indices = np.indices(letters.shape).reshape((letters.ndim, -1)).T

classifier = RadiusNeighborsClassifier(0)
from sklearn.neighbors import KNeighborsClassifier

mask = subjects == 0
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(
    X[mask], y[mask], indices[mask],
    test_size=0.25,
    stratify=y[mask]
)

classifier = KNeighborsClassifier(9)

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
find_strokes(letters[0, 9, 3], 2, dist=11, db=-9)

idx = np.where(y_train == evaluate.alphabet.index('O'))[0]
index_train[idx]
find_strokes(letters[0, 14, 5], verbose=1)
X_train[idx].astype(int)

find_strokes(letters[0, 0, 0], verbose=1)

idx = np.where(y_train == evaluate.alphabet.index('A'))[0]
index_train[idx]

f, (a, b, c) = plt.subplots(3)
X = mlab.specgram(np.diff(letters[0, 4, 14]), NFFT=512, noverlap=511)[0]
a.imshow(np.log(X), origin='lower', aspect='auto')
b.plot(np.log(X)[:120].sum(axis=0))
summed = X[:120].sum(axis=0)
summed /= np.max(summed)
c.plot(np.log(summed))
f.show()

find_strokes(letters[0, 0, 0], verbose=2, db=-8)
X_train[idx].astype(int)

proba = classifier.predict_proba([[0, 1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]])[0]
list(zip(evaluate.alphabet, proba))
