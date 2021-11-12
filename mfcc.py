from evaluate import load_dataset
import librosa
import numpy as np
from dtw import dtw


letters, fs = load_dataset()
a = letters[0, 0, 0].astype(float) / np.iinfo(np.int32).max

import matplotlib.pyplot as plt
librosa.feature.mfcc(a).shape
plt.specgram(a)
plt.imshow(librosa.feature.mfcc(a), origin='lower')

mfccs = None
def preprocessor(letters, fs):
    global mfccs
    mfccs = np.empty(letters.shape, dtype=object)
    for i, letter in enumerate(letters):
        letter = letter.astype(float) / np.iinfo(np.int32).max
        mfccs[i] = librosa.feature.mfcc(letter).T
    return np.arange(len(mfccs)).reshape(-1, 1)

def dtw_dist(a, b):
    a = mfccs[int(a[0])]
    b = mfccs[int(b[0])]
    return dtw(a, b, dist_method="cosine").normalizedDistance

from sklearn.neighbors import KNeighborsClassifier

cls = KNeighborsClassifier(1, metric=dtw_dist, algorithm='brute', n_jobs=8)

X = preprocessor(letters.reshape(-1), fs)
y = np.indices(letters.shape)[1].reshape(-1)
subjects = np.indices(letters.shape)[0].reshape(-1)
indices = np.indices(letters.shape).reshape((letters.ndim, -1)).T

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
mask = subjects == 0
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(
    X[mask], y[mask], indices[mask],
    test_size=0.25,
    stratify=y[mask]
)

cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
(y_pred == y_test).mean()

import evaluate
evaluate.run(preprocessor, cls)

# Single-subject accuracy (0): 59.23%
# Single-subject accuracy (1): 63.08%
# Single-subject accuracy (2): 76.92%
# Single-subject accuracy (3): 67.69%
# All-subject accuracy: 57.5%
# - Subject 0: 71.65%
# - Subject 1: 76.67%
# - Subject 2: 47.95%
# - Subject 3: 36.22%
# Left-out-subject accuracy (0): 6.92%
# Left-out-subject accuracy (1): 7.12%
# Left-out-subject accuracy (2): 4.81%
# Left-out-subject accuracy (3): 5.19%
