import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


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
                    s = sigmoid(y[i] * np.dot(w,Xe[i]))
                    e += s*a
        gt = (-1.0/batch_size)*e
        hw.append(w)
        w += learning_rate*(-gt)
    return w


def logistic_predict(X,w):
    Xe = np.insert(X,0,1.0)
    s = np.dot(w, Xe)
    return sigmoid(s)


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def run_tests1():
    # Teste para pontos em duas dimensões
    N = 1000

    mean_pos = [0, 0]
    mean_neg = [4, 0]
    cov = [
    [1, 0],
    [0, 1]
    ]

    # Exemplos positivos
    X_pos = np.random.multivariate_normal(mean_pos, cov, size = N//2)
    Y_pos = np.zeros(N//2) + 1

    # Exemplos negativos
    X_neg = np.random.multivariate_normal(mean_neg, cov, size = N-N//2)
    Y_neg = np.zeros(N-N//2) - 1

    # Dataset completo
    X = np.concatenate([X_pos, X_neg], axis = 0)
    Y = np.concatenate([Y_pos, Y_neg], axis = 0)

    # Embaralha os dados
    perm = np.random.permutation(N)
    X = X[perm]
    Y = Y[perm]

    # Visualização dos dados para comparação com a regressão logística
    f, ax = plt.subplots(1,2)
    f.set_size_inches(8,8)
    f.subplots_adjust(hspace=20.0)
    ax[0].scatter(X[:,0], X[:,1], c=['green' if y > 0 else 'orange' for y in Y])

    # Regressão logística e visualização da classificação
    w = logistic_fit(X, Y)
    print(f'The predicted weights were: ' + str(w))
    colors = ['blue' if logistic_predict(X[i],w) < 0.5 else 'red' for i in range(X.shape[0])]
    ax[1].scatter(X[:,0], X[:,1], color=colors)
    plt.show()


if __name__ == '__main__':
    run_tests1()

