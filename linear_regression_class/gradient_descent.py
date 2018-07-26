# notes for this course can be found at:
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# https://www.udemy.com/data-science-linear-regression-in-python
# http://ml-cheatsheet.readthedocs.io/en/latest/

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3
X = np.zeros((N, D))
X[:, 0] = 1  # bias term
X[:5, 1] = 1
X[5:, 2] = 1
Y = np.array([0] * 5 + [1] * 5)

# print X so you know what it looks like
print("X:", X)

# won't work!
# w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

# let's try gradient descent
costs = []  # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D)  # randomly initialize w
print('weights initial: ')
print(w)
learning_rate = 0.001
# Take 1000 steps
for t in range(1000):
    # update w
    Yhat = X.dot(w)
    print('Yhat now')
    print(Yhat.T)
    delta = Yhat - Y
    print('delta now')
    print(delta.T)
    w = w - learning_rate * X.T.dot(delta)
    print('weights now')
    print(w.T)

    # find and store the cost
    mse = delta.dot(delta) / N
    print('mse now')
    print(mse)
    print('\n')
    costs.append(mse)

# plot the costs
plt.plot(costs)
plt.show()

print("final w:", w)

# plot prediction vs target
plt.plot(Yhat, label='prediction')
plt.plot(Y, label='target')
plt.legend()
plt.show()
