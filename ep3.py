import numpy as np


def logistic_fit(X, y, w = None, batch_size=None, learning_rate=1e-2,
        num_iterations=1000, return_history=False):
    hw = []
    if w == None:
        w = np.asarray(np.random.rand(X.shape[1] + 1))
    if batch_size == None or batch_size > X.shape[0]:
        batch_size = X.shape[0]

    # Cross-Entropy 
    Xe = np.hstack((np.ones((X.shape[0],1)),X))
    for t in range(num_iterations):
        e = 0.0
        for i in range(batch_size):
                    a = -y[i]*Xe[i]
                    b = y[i] * np.dot(w,Xe[i])
                    s = 1.0 / (1.0 + np.exp(-b)) # sigmoid function 
                    e += s*a
        gt = (-1.0/batch_size)*e
        hw.append(w)
        w += learning_rate*(-gt)

    if return_history == False:
        return w
    else:
        return w, hw


def logistic_predict(X,w):
    p = []
    Xe = np.hstack((np.ones((X.shape[0],1)),X))
    for i in range(Xe.shape[0]):
        s = np.dot(w, Xe[i])
        p.append(1.0 / (1.0 + np.exp(-s)))
    return p
