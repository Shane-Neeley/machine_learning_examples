import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt

def train_linreg(Xtrain, Ytrain, eta, epochs):

    costs = []
    eta0 = T.dscalar('eta0')
    y = T.dvector(name='y')
    X = T.dmatrix(name='X')

    w = theano.shared(np.zeros(
        shape=(Xtrain.shape[1] + 1),
        dtype=theano.config.floatX),
        name='w')

    net_input = T.dot(X, w[1:]) + w[0]
    errors = y - net_input

    cost = T.sum(T.pow(errors, 2))

    gradient = T.grad(cost, wrt=w)
    update = [(w, w - eta0 * gradient)]

    train = theano.function(
        inputs = [eta0], # learning rate here?
        outputs = cost,
        updates = update,
        givens = {
            X: Xtrain, y: Ytrain
        }
    )

    for _ in range(epochs):
        costs.append(train(eta))

    return costs, w

###################

N = 10
Xtrain = np.array([[i] for i in np.arange(N)], dtype=theano.config.floatX)
def func(x):
    return x*(5+np.random.randn(1)[0]) + 10
Ytrain = np.array([func(i) for i in Xtrain], dtype=theano.config.floatX).flatten()
print(Ytrain)

plt.scatter(Xtrain, Ytrain)
plt.show()

if  __name__ == '__main__':
    costs, w = train_linreg(Xtrain, Ytrain, 0.001, 100)
    print(w.get_value())
    plt.plot(range(1, len(costs) + 1), costs)
    plt.show()
