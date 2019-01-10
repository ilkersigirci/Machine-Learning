import numpy as np


def viterbi(transitionPath, estimatePath, observationPath):
    
    transitionMatrix = np.loadtxt(transitionPath,dtype = int)

    return transitionMatrix


if __name__ == "__main__":
    transitionPath = "/mnt/d/DERS/0000GITHUB/Machine-Learning/THE3/transition_matrix.txt"
    estimatePath = "/mnt/d/DERS/0000GITHUB/Machine-Learning/THE3/estimate_matrix.txt"
    observationPath = "/mnt/d/DERS/0000GITHUB/Machine-Learning/THE3/observations.txt"
    print viterbi(transitionPath, estimatePath, observationPath)