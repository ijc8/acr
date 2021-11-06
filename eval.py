import numpy as np
import scipy.io.wavfile
import scipy.signal

from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline


alphabet = ''.join(chr(ord("A") + i) for i in range(26))


def load_dataset():
    letters = np.empty((4, 26, 20), dtype=object)
    for i, j, k in np.ndindex(letters.shape):
        subject = i + 1
        letter = alphabet[j]
        fs, x = scipy.io.wavfile.read(f"letters/{subject}/{letter}/PZ/{k}.wav")
        letters[i, j, k] = x
    # Assumes all letters have same sample rate.
    return letters, fs


def eval_accuracy(extractor, classifier, test, train):
    testY = np.indices(test.shape[:3])[1].reshape(-1)
    trainY = np.indices(train.shape[:3])[1].reshape(-1)
    test = test.reshape(-1)
    train = train.reshape(-1)
    features = extractor(np.hstack((test, train)))
    testX, trainX = features[:len(test)], features[len(test):]
    classifier.fit(trainX, trainY)
    predictY = classifier.predict(testX)
    return np.mean(testY == predictY)


def run(preprocessor, classifier, seed=None):
    # TODO: Add option to plot confusion matrices, in addition to printing accuracy.
    letters, fs = load_dataset()

    # Avoid re-doing preprocessing for each of these tasks.
    X = preprocessor(letters.reshape(-1), fs)
    y = np.indices(letters.shape[:3])[1].reshape(-1)
    subjects = np.indices(letters.shape[:3])[0].reshape(-1)

    # Single-subject accuracy (fully personalized mode)
    for subject in range(len(letters)):
        mask = subjects == subject
        X_train, X_test, y_train, y_test = train_test_split(
            X[mask], y[mask], test_size=0.25,
            stratify=y[mask], random_state=seed
        )
        classifier.fit(X_train, y_train)
        print(f"Single-subject accuracy ({subject}): {round(classifier.score(X_test, y_test) * 100, 2)}%")

    # All-subjects accuracy (semi-personalized mode)
    X_train, X_test, y_train, y_test, subjects_train, subjects_test = train_test_split(
        X, y, subjects, test_size=0.25,
        stratify=y, random_state=seed
    )
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    print(f"All-subject accuracy: {round((predicted == y_test).mean() * 100, 2)}%")
    # Break down accuracy by subject
    for subject in range(len(letters)):
        mask = subjects_test == subject
        print(f"- Subject {subject}: {round((predicted[mask] == y_test[mask]).mean() * 100, 2)}%")

    # Left-out-subjects accuracy (new user mode)
    logo = LeaveOneGroupOut()
    for subject, (train_index, test_index) in enumerate(logo.split(X, y, subjects)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        print(f"Left-out-subject accuracy ({subject}): {round(classifier.score(X_test, y_test) * 100, 2)}%")
