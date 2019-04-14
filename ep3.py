import matplotlib.pyplot as plt
import numpy as np

def logistic_fit(X, y, w = None, batch_size=None, learnig_rate=1e-2,
        num_iterations=1000, return_history=False):
    hw = []
    if w == None:
        w = np.random.rand(1, X.shape[1] + 1)
    if batch_size == None or batch_size > X.shape[0]:
        batch_size = X.shape[0]

   # Cross-Entropy 
   X = np.hstack((np.ones((X.shape[0],X))))
    for t in range(num_iterations):
        Ein = 0
        for i in range(batch_size):
            Ein += np.log(1 + np.exp(-y[i] * np.dot(w, X[i:])))
        vt = (1.0/batch_size)*Ein
        hw.append(w)
        w += learning_rate*(vt)
    return w if return_history == False else w, hw

def logistic_predict(X,w):
    X.np.hstack((np.ones((X.shape[0],X))))
    s = np.dot(X, w)
    return np.exp(s) / (1 + np.exp(s))

def run_tests():
    return 0




if __name__ == '__main__':
    run_tests()

