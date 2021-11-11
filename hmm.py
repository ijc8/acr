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
        # NOTE: This heavily favors fewer-letter explanations, because the within-letter
        # transition probabilties are 1, while the between-letter probabilities are
        # necessarily < 1 (uniform 1/n_sequences). This is a problem.
        # I believe this could be resolved by modifying the probabilities of the start states
        # to be inversely exponentially proportional with the length of the sequence,
        # so that the long letters receive the same probability-penalty as the short ones over time.
        # This involves solving a polynomial of degree <max sequence length + 1>,
        # which could be done with `np.roots`.
        # However, it may be simpler (and useful for other reasons) to just customize Viterbi.
        n_sequences = len(X)
        n_states = sum(len(span) for span in X) + 2
        n_features = X[0].shape[1]
        startprob = np.zeros(n_states)
        transmat = np.zeros((n_states, n_states))
        means = np.zeros((n_states, n_features))
        covars = np.ones((n_states, n_features))

        # >= 0 indicates start of a sequence.
        # < 0 indicates continuing a sequence.
        states = np.zeros(n_states)

        start_states = np.empty(n_sequences, dtype=int)
        end_states = np.empty(n_sequences, dtype=int)
        state = 0
        # State 0 represents the gap between letters.
        # At the end, we'll connect it to all the start nodes.
        states[state] = np.nan
        startprob[state] = 0
        means[state] = [0.5, 0]
        state += 1
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
            # Link the last node in the sequence to the gap state.
            transmat[state-1, 0] = 1
            end_states[i] = state - 1

        # Link the gap state to all the start nodes.
        transmat[0, start_states] = 1 / (n_sequences + 1)
        # Link the gap state to the end node.
        transmat[0, state] = 1 / (n_sequences + 1)
        # Construct the end node; make it unlikely that the end node is mixed up with any other node.
        states[state] = np.nan
        means[state] = 1e6
        transmat[state, state] = 1

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
            # Add gap state + end state to the end.
            sample = np.vstack((sample, [[0, 0], [1e6, 1e6]]))
            prob, seq = self.model.decode(sample)
            states = self.states[seq]
            print(states)
            start_states = states[~np.signbit(states)].astype(int)
            if len(start_states) > 3:
                print(f"Warning: matched multi-letter sequence: {start_states}")
            y[i] = start_states[0]
        return y

from sklearn.model_selection import train_test_split

import scipy
from scipy import signal, stats

def emission(template, spans):
    log_prob = 0
    for expected, observed in zip(template, spans):
        p = stats.norm.pdf(observed, loc=expected, scale=[50, 1])
        log_prob += np.sum(np.log(p))
    return log_prob

def custom_viterbi(templates, spans):
    # Customized implementation of Viterbi algorithm.
    # Lacks transition probabilities (and therefore does not penalize sequences with more characters.)
    # Efficiently handles fixed subsequences (where internal transition probabilities are all 1).
    # TODO: Allow gap between letters. (Maybe just an extra one-span template?)
    # This could be further optimized by combining all sequences of the same length,
    # so that there would be a very small (max(len, templates)) number of states in the trellis.
    n_states = len(templates)
    trellis = np.ones((len(spans), n_states)) * -np.inf
    pointers = np.zeros((len(spans), n_states, 2), dtype=int)

    for s, template in enumerate(templates):
        if len(template) > len(spans):
            continue
        # Skip ahead to the end of the template.
        # (Intermediate probabilities will be 0, reflecting that you can't switch in the middle of a letter.)
        trellis[len(template) - 1, s] = emission(template, spans[:len(template)])

    for o in range(1, len(spans)):
        for s in range(n_states):
            template = templates[s]
            if o + len(template) > len(spans):
                # Not enough observations left for this template.
                continue
            emission_log_prob = emission(template, spans[o:o + len(template)])
            candidates = np.zeros(n_states)
            for k in range(n_states):
                prev_log_prob = trellis[o-1, k]
                candidates[k] = prev_log_prob + emission_log_prob
            k = np.argmax(candidates)
            trellis[o + len(template) - 1, s] = candidates[k]
            pointers[o + len(template) - 1, s] = [o-1, k]

    o = len(spans) - 1
    best_prob = np.max(trellis[o])
    k = np.argmax(trellis[o])
    start = o - len(templates[k]) + 1
    best_path = [(start, o + 1, k)]
    while start > 0:
        o, k = pointers[o, k]
        start = o - len(templates[k]) + 1
        best_path.insert(0, (start, o + 1, k))

    return best_path, best_prob

extra = np.array([None])
extra[0] = np.array([[30, 0]])

templates = np.concatenate((X_train, extra))
correspondence = ''.join(evaluate.alphabet[c] for c in y_train) + '|'
''.join(correspondence[i] for s, e, i in custom_viterbi(templates, the_spans * [1, 1])[0])

''.join(correspondence[i] for s, e, i in custom_viterbi(templates, quick_spans)[0])
quick_spans
pred_spans
pred_spans = np.vstack([templates[i] for _, _, i in custom_viterbi(templates, quick_spans)[0]])

evaluate.alphabet[7]
y_train[127]
y_train[192]
y_train[380]

evaluate.alphabet[2]

templates = X_train
y_train[263]
y_test[0]
spans = X_test[0]
X_test[0]
custom_viterbi(templates, spans)

custom_viterbi(templates, np.vstack((X_test[0], X_test[3])))
y_pred = np.zeros(len(y_test))
for i, spans in enumerate(X_test):
    path = custom_viterbi(templates, spans)
    print(path)
    y_pred[i] = y_train[path[-1][-1]]
print((y_pred == y_test).mean())
np.where(y_pred == y_test)


fs, the = scipy.io.wavfile.read("words/the.wav")
import matplotlib.pyplot as plt
plt.specgram(the, NFFT=512, noverlap=256)
the_spans = extract_spans(the, verbose=1, db=-13, normalize_time=False)[:, 1:]
len(the_spans)
the_spans * [3, 1]
cls.predict([the_spans * [1, 1]])
evaluate.alphabet[15]

fs, quick = scipy.io.wavfile.read("words/quick.wav")
import matplotlib.pyplot as plt
plt.specgram(quick, NFFT=512, noverlap=256)
quick_spans = extract_spans(quick, verbose=1, normalize_time=False)[:, 1:]
quick_spans[:3]
cls.predict([quick_spans[:3]])
dir(cls.model)
X_train[y_train == 16][1]
index_train[y_train == 25][1]
X_train[y_train == 25][1]
evaluate.alphabet.index('Q')
evaluate.alphabet[25]

extract_spans(letters[0,25,5], verbose=1)

# 6
len(extract_spans(letters[0, 17, 20], verbose=1))

letters, fs = evaluate.load_dataset()
# NOTE: We skip the first feature (start time), because the HMM considers each stroke independently.
X = np.array([extract_spans(letter, normalize_time=False)[:, 1:] for letter in letters.reshape(-1)], dtype=object)
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
X[indices.tolist().index([0, 0, 0])]
# cls.predict([np.vstack((X[indices.tolist().index([0, 0, 0])], [[0, 0], [1e6, 1e6]]))])
cls.predict([X[indices.tolist().index([0, 0, 0])]])
X[0]
cls.predict([X[indices.tolist().index([0, 1, 0])]])

concat = np.concatenate((letters[0,4,0], letters[0, 2, 0]))

concat = np.concatenate((letters[0,4,0], letters[0, 1, 1]))
concat_spans = extract_spans(concat, verbose=1)[:, 1:]

concat_spans = np.vstack((extract_spans(letters[0,4,0])[:, 1:], [[0.5, 0]], extract_spans(letters[0,4,0])[:, 1:]))

cls.predict([concat_spans])
concat_spans[:, 0] *= 2
concat_spans[:, 0].sum()

len(X[(y == 19) & (subjects == 0)])
list(map(len, X[(y == 4) & (subjects == 0)]))
len(concat_spans)

evaluate.alphabet[4]
evaluate.alphabet[22]
extract_spans(letters[0, 24, 3], verbose=1)

cls.predict([concat_spans])

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
