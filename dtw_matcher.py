import numpy as np

## A noisy sine wave as query
idx = np.linspace(0,6.28,num=100)
query = np.sin(idx) + np.random.uniform(size=100)/10.0

## A cosine is for template; sin and cos are offset by 25 samples
template = np.cos(idx)

## Find the best match with the canonical recursion formula
from dtw import *
help(dtw)
alignment = dtw(query, template, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(query, template, keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
rabinerJuangStepPattern(6,"c").plot()

## And much more!


from evaluate import load_dataset
letters, fs = load_dataset()

from matplotlib import mlab
from scipy import signal

a = mlab.specgram(np.diff(letters[0,0,0]), NFFT=512, noverlap=256)[0].T
b = mlab.specgram(np.diff(letters[0,0,1]), NFFT=512, noverlap=256)[0].T
a.shape
b.shape
max(signal.correlate(a, b, mode='valid')) / max(len(a), len(b))

from fastdtw import fastdtw
distance, path = fastdtw(a, b, dist=scipy.spatial.distance.cosine)
distance
path

alignment = dtw(a, b, keep_internals=True, dist_method="cosine")
alignment.plot(type="threeway")
alignment.distance
help(alignment)
alignment.normalizedDistance
help(alignment.plot)
help(dtwPlotTwoWay)
dtwPlotTwoWay(alignment, ts_type="image")

def preprocessor(letters, fs):
    global specgrams
    specgrams = np.array([
        mlab.specgram(np.diff(letter), NFFT=512, noverlap=0)[0].T # .sum(axis=1)
        for letter in letters
    ], dtype=object)
    return np.arange(len(specgrams)).reshape(-1, 1)

def dtw_dist(a, b):
    a = specgrams[int(a[0])]
    b = specgrams[int(b[0])]
    return dtw(a, b, dist_method="cosine").normalizedDistance
    # return dtw(a, b).normalizedDistance
    # despite the name, this seems to be much slower...
    # return fastdtw(a, b, dist=scipy.spatial.distance.cosine)[0]

def cc_dist(a, b):
    a = specgrams[int(a[0])]
    b = specgrams[int(b[0])]
    return np.max(signal.correlate(a, b, mode='valid')) / max(len(a), len(b))


cls = KNeighborsClassifier(1, metric=distance, algorithm='brute', n_jobs=8)

cls = KNeighborsClassifier(1, metric=cc_dist, algorithm='brute')

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
cls.predict([X_test[0]])
print(y_test[0])
y_pred = cls.predict(X_test)
print((y_pred == y_test).mean())
