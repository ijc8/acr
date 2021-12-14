import matplotlib.pyplot as plt
import numpy as np
import pickle

with open("by_num_classes.pkl", "rb") as f:
    by_num_classes = pickle.load(f)

by_num_classes = np.array(by_num_classes)
means = by_num_classes.mean(axis=2)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(26) + 1, means, label=["Single-subject", "All-subjects", "Left-out"])
plt.xlim(1, 26)
plt.ylim(0, 1.01)
plt.xticks(np.arange(1, 27))
plt.legend()
plt.title("Letter accuracy vs. number of classes (averaged across subjects)")
plt.ylabel("Accuracy")
plt.xlabel("Number of classes (1 = A, 2 = AB, 3 = ABC...)")
plt.savefig("plots/by_num_classes.png", facecolor="white", dpi=150)
