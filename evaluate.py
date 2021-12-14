import numpy as np
import scipy.io.wavfile
import scipy.signal

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline


alphabet = ''.join(chr(ord("A") + i) for i in range(26))


def load_dataset():
    letters = np.empty((4, 26, 20), dtype=object)
    for i, j, k in np.ndindex(letters.shape):
        subject = i + 1
        letter = alphabet[j]
        fs, x = scipy.io.wavfile.read(f"letters/{subject}/{letter}/PZ/{k}.wav")
        letters[i, j, k] = x.astype(float) / np.iinfo(np.int32).max
    # Assumes all letters have same sample rate.
    return letters, fs

def run(preprocessor, classifier, subset=None, seed=None, plot=True):
    letters, fs = load_dataset()
    labels = alphabet
    if subset is not None:
        letters = letters[:, subset, :]
        labels = np.array(list(alphabet))[subset]

    # Avoid re-doing preprocessing for each of these tasks.
    X = preprocessor(letters.reshape(-1), fs)
    y = np.indices(letters.shape)[1].reshape(-1)
    subjects = np.indices(letters.shape)[0].reshape(-1)

    results = np.empty((3, len(letters)))

    # Single-subject accuracy (fully personalized mode)
    for subject in range(len(letters)):
        mask = subjects == subject
        X_train, X_test, y_train, y_test = train_test_split(
            X[mask], y[mask], test_size=0.25,
            stratify=y[mask], random_state=seed
        )
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        results[0, subject] = (y_pred == y_test).mean()
        print(f"Single-subject accuracy ({subject}): {round((y_pred == y_test).mean() * 100, 2)}%")
        if plot:
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels, include_values=False)

    # All-subjects accuracy (semi-personalized mode)
    X_train, X_test, y_train, y_test, subjects_train, subjects_test = train_test_split(
        X, y, subjects, test_size=0.25,
        stratify=y, random_state=seed
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f"All-subject accuracy: {round((y_pred == y_test).mean() * 100, 2)}%")
    if plot:
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels, include_values=False)
    # Break down accuracy by subject
    for subject in range(len(letters)):
        mask = subjects_test == subject
        results[1, subject] = (y_pred[mask] == y_test[mask]).mean()
        print(f"- Subject {subject}: {round((y_pred[mask] == y_test[mask]).mean() * 100, 2)}%")
        # ConfusionMatrixDisplay.from_predictions(y_test[mask], y_pred[mask], display_labels=labels, include_values=False)

    # Left-out-subjects accuracy (new user mode)
    logo = LeaveOneGroupOut()
    for subject, (train_index, test_index) in enumerate(logo.split(X, y, subjects)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        results[2, subject] = (y_pred == y_test).mean()
        print(f"Left-out-subject accuracy ({subject}): {round((y_pred == y_test).mean() * 100, 2)}%")
        if plot:
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels, include_values=False)

    return results
