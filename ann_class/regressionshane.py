
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle


# In[25]:


def saddle(x1, x2):
    return x1 * x2

N = 500
X = np.random.random((N, 2)) * 4 - 2  # in between (-2, +2)
Y = X[:, 0] * X[:, 1]  # makes a saddle shape


# In[26]:


D = 2  #p number of samples
M = 100  # num hidden units


# In[27]:


# layer 1
W = np.random.randn(D, M) / np.sqrt(D)
b = np.zeros(M)

# layer 2
V = np.random.randn(M) / np.sqrt(M)
c = 0


# In[28]:


# how to get the output
# consider the params global
def forward(X):
    Z = X.dot(W) + b
    Z = Z * (Z > 0)  # relu
    # Z = np.tanh(Z)

    Yhat = Z.dot(V) + c
    return Z, Yhat


# how to train the params
def derivative_V(Z, Y, Yhat):
    return (Y - Yhat).dot(Z)


def derivative_c(Y, Yhat):
    return (Y - Yhat).sum()


def derivative_W(X, Z, Y, Yhat, V):
    # dZ = np.outer(Y - Yhat, V) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(Y - Yhat, V) * (Z > 0)  # relu
    return X.T.dot(dZ)


def derivative_b(Z, Y, Yhat, V):
    # dZ = np.outer(Y - Yhat, V) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(Y - Yhat, V) * (Z > 0)  # this is for relu activation
    return dZ.sum(axis=0)


def update(X, Z, Y, Yhat, W, b, V, c, learning_rate=1e-4):
    gV = derivative_V(Z, Y, Yhat)
    gc = derivative_c(Y, Yhat)
    gW = derivative_W(X, Z, Y, Yhat, V)
    gb = derivative_b(Z, Y, Yhat, V)

    V += learning_rate * gV
    c += learning_rate * gc
    W += learning_rate * gW
    b += learning_rate * gb

    return W, b, V, c


# so we can plot the costs later
def get_cost(Y, Yhat):
    return ((Y - Yhat)**2).mean()


# In[30]:


costs = []
for i in range(2000):
    Z, Yhat = forward(X)
    W, b, V, c = update(X, Z, Y, Yhat, W, b, V, c)
    cost = get_cost(Y, Yhat)
    costs.append(cost)
    if i % 25 == 0:
        print(cost)


# In[31]:


plt.plot(costs)
plt.show()
