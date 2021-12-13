import dtw_matcher
import numpy as np

def main():
    runs = 10
    accuracies = np.zeros(runs)
    for i in range(runs):
        accuracies[i] = dtw_matcher.main()
    print("mean: "+str(np.mean(accuracies)))
    print("std:" +str(np.std(accuracies)))

    return

if __name__ == '__main__':
    main()