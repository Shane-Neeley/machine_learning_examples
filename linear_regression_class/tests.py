import numpy as np
import matplotlib.pyplot as plt

N = 50

X = np.linspace(0, 10, N)
# print(X)
Y = 0.5*X + np.random.randn(N)
# print(Y)

Y[-1] += 30
Y[-2] += 30

plt.scatter(X,Y)
plt.show()

X = np.vstack([np.ones(N), X]).T
# print(X)

w_ml = np.linalg.solve(X.T.dot(X),X.T.dot(Y))

print(w_ml)

Yhat_ml = X.dot(w_ml)
print(X)
print(X[:,1])
print(np.eye(2))

plt.scatter(X[:,1], Y)
plt.scatter(X[:,1], Yhat_ml)
plt.show()
