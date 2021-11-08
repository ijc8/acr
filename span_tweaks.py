
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
extract_spans(letters[0,0,1], verbose=1)

indices.tolist().index([0, evaluate.alphabet.index("Q"), 0])

# Limits of spans: observe how this X (0) and this T (1) look identical.
# After analyzing spectrograms, it seems that these look identical
# because the cutoff is so low (:25).
# More spectral information (even just in the sum) may help distinguish them.
# Or we can bring in e.g. the spectral centroid.
# I think the key here is observing that the cutoff is good for avoiding
# non-strokes, but we need not apply it when collecting stroke info.
# (This realization made the single-subject classification jump up > 10 percentage points)
alt_dist_metric([460], [381], verbose=True)

alt_dist_metric([320], [321], verbose=True)


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

index_train[y_train == evaluate.alphabet.index('L')]
index_test[y_test == evaluate.alphabet.index('L')]
indices.tolist().index([0, evaluate.alphabet.index('L'), 0])
indices[220]
indices[221]
extract_spans(letters[0, 11, 0], verbose=1, db=-11)
extract_spans(letters[0, 11, 1], verbose=1, db=-11, cutoff=22)
alt_dist_metric([220], [221], verbose=True)

index_train[y_train == evaluate.alphabet.index('M')]
index_test[y_test == evaluate.alphabet.index('M')]
indices[240]
indices[247]
extract_spans(letters[0, 12, 0], verbose=1, cutoff=25, db=-8)
extract_spans(letters[0, 12, 10], verbose=1, cutoff=25, db=-5)
alt_dist_metric([240], [250], verbose=True)

indices[182]
indices[188]
extract_spans(letters[0, 9, 2], verbose=1)
extract_spans(letters[0, 9, 8], verbose=1)
alt_dist_metric([182], [188], verbose=True)


index_train[y_train == evaluate.alphabet.index('N')]
index_test[y_test == evaluate.alphabet.index('N')]
indices[260]
indices[262]
extract_spans(letters[0, 13, 0], verbose=1, cutoff=25, db=-8)
extract_spans(letters[0, 13, 2], verbose=1, cutoff=25, db=-8)
alt_dist_metric([260], [262], verbose=True)

evaluate.alphabet[18]
index_train[y_train == evaluate.alphabet.index('Q')]
index_test[y_test == evaluate.alphabet.index('Q')]
classifier.predict(X_test[y_test == evaluate.alphabet.index('Q')])
extract_spans(letters[0, 16, 11], verbose=1, db=-11)
extract_spans(letters[0, 16, 18], verbose=1)
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
extract_spans(letters[0, 9, 3], verbose=2)

idx = np.where(y_train == evaluate.alphabet.index('O'))[0]
index_train[idx]
extract_spans(letters[0, 14, 5], verbose=1)
X_train[idx].astype(int)

plt.specgram(letters[1, 8, 14], NFFT=512, noverlap=256);
extract_spans(letters[1, 10, 1], verbose=1)

idx = np.where(y_train == evaluate.alphabet.index('A'))[0]
index_train[idx]

evaluate.alphabet.index("T")

f, (a, b, c) = plt.subplots(3, figsize=(20, 20), sharex=True)
X = mlab.specgram(np.diff(letters[0, 9, 8]), NFFT=512, noverlap=256)[0]
# X -= np.median(X, axis=1)[:, None]
a.imshow(np.log(X), origin='lower', aspect='auto')
summed = X[:22].sum(axis=0)
# summed = X.sum(axis=0)
summed = signal.medfilt(summed, 11)
summed /= np.max(summed)
b.plot(summed)
b.axhline(10**(-12/10))
c.plot(np.log(summed))
f.show()



extract_spans(letters[1, 10, 1], verbose=1, cutoff=120)
X_train[idx].astype(int)

proba = classifier.predict_proba([[0, 1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]])[0]
list(zip(evaluate.alphabet, proba))
