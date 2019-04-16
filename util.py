import matplotlib.pyplot as plt
import numpy as np
from ep3 import logistic_fit, logistic_predict

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
    ax[0].set_title('Estimativa esperada')

    # Regressão logística e visualização da classificação
    w = logistic_fit(X, Y)
    print(f'The predicted weights were: ' + str(w))
    pred = logistic_predict(X,w)
    colors = ['blue' if pred[i] < 0.5 else 'red' for i in range(X.shape[0])]
    ax[1].scatter(X[:,0], X[:,1], color=colors)
    ax[1].set_title('Regressão Logística')
    plt.show()

def run_tests2():

    N = 1000

    mean_pos = [0, 0, 0, 0]
    mean_neg = [4, 0, 1, 0]
    # Calculando a covariancia com variância fixa em 0.5
    cov = np.dot(.5, np.eye(4,4))

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

    w = logistic_fit(X, Y)
    print('Weights: ' + str(w))
    pred = logistic_predict(X,w)


if __name__ == '__main__':
    run_tests1()
    run_tests2()

