import numpy as np

from hmmlearn import hmm

import evaluate
from spans import extract_spans


class HMMLetterClassifier:
    def fit(self, X, y):
        # One state *sequence* per letter-spans in X; one state per span within X.
        # TODO: Later, we may want to consolidate versions of the same letter
        # with the same number of spans, in order to reduce the size of the HMM.
        # TODO: This currently uses the durations reported by `extract_spans`,
        # which are normalized to the letter length. This will likely cause problems
        # when we try to use this on whole words.
        n_sequences = len(X)
        n_states = sum(len(span) for span in X)
        n_features = X[0].shape[1]
        startprob = np.zeros(n_states)
        transmat = np.zeros((n_states, n_states))
        means = np.zeros((n_states, n_features))
        covars = np.ones((n_states, n_features))

        # >= 0 indicates start of a sequence.
        # < 0 indicates continuing a sequence.
        states = np.ones(n_states) * -1

        start_states = np.empty(n_sequences, dtype=int)
        end_states = np.empty(n_sequences, dtype=int)
        state = 0
        for i, (spans, label) in enumerate(zip(X, y)):
            # Create a state sequence to represent this particular version of a letter.
            start_states[i] = state
            states[state] = label
            startprob[state] = 1 / n_sequences
            means[state] = spans[0]
            state += 1
            for span in spans[1:]:
                states[state] = -float(label)
                means[state] = span
                # Link nodes in the state sequence together with 100% probability.
                transmat[state-1, state] = 1
                state += 1
            # After this loop, we'll link the last node in the sequence with all the start nodes.
            end_states[i] = state - 1

        for end_state in end_states:
            transmat[end_state, start_states] = 1 / n_sequences

        model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag")
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = covars

        self.states = states
        self.model = model

    def predict(self, X):
        y = np.empty(len(X), dtype=int)
        for i, sample in enumerate(X):
            prob, seq = self.model.decode(sample)
            states = self.states[seq]
            start_states = states[~np.signbit(states)].astype(int)
            if len(start_states) > 1:
                print(f"Warning: matched multi-letter sequence: {start_states}")
            y[i] = start_states[0]
        return y

from sklearn.model_selection import train_test_split


letters, fs = load_dataset()
# NOTE: We skip the first feature (start time), because the HMM considers each stroke independently.
X = np.array([extract_spans(letter)[:, 1:] for letter in letters.reshape(-1)], dtype=object)
y = np.indices(letters.shape)[1].reshape(-1)

subjects = np.indices(letters.shape)[0].reshape(-1)
indices = np.indices(letters.shape).reshape((letters.ndim, -1)).T

mask = subjects == 0
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(
    X[mask], y[mask], indices[mask],
    test_size=0.25,
    stratify=y[mask]
)

cls = HMMLetterClassifier()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
print(f"{round((y_test == y_pred).mean() * 100, 2)}%")


def span_durations_and_amps(letters, fs):
    return np.array([extract_spans(letter)[:, 1:] for letter in letters], dtype=object)
evaluate.run(span_durations_and_amps, HMMLetterClassifier())

# Output:
# Single-subject accuracy (0): 40.0%
# Single-subject accuracy (1): 33.85%
# Single-subject accuracy (2): 43.85%
# Single-subject accuracy (3): 40.77%
# All-subject accuracy: 36.92%
# - Subject 0: 33.59%
# - Subject 1: 41.27%
# - Subject 2: 38.97%
# - Subject 3: 33.85%
# Warning: matched multi-letter sequence: [18  4]
# Left-out-subject accuracy (0): 13.46%
# Left-out-subject accuracy (1): 14.62%
# Left-out-subject accuracy (2): 18.08%
# Left-out-subject accuracy (3): 16.92%
