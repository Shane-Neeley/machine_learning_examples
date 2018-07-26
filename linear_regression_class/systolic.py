# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# need to sudo pip install xlrd to use pd.read_excel
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel('mlr02.xls') # switch to anaconda for this to work
X = df.as_matrix()

# using age to predict systolic blood pressure
plt.scatter(X[:,1], X[:,0])
# plt.show()
# looks pretty linear!

# using weight to predict systolic blood pressure
plt.scatter(X[:,2], X[:,0])
# plt.show()
# looks pretty linear!

df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]
X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X, Y):
    w = np.linalg.solve( X.T.dot(X), X.T.dot(Y) )
    Yhat = X.dot(w)

    # determine how good the model is by computing the r-squared
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2

print("r2 for x2 only:", get_r2(X2only, Y))
print("r2 for x3 only:", get_r2(X3only, Y))
print("r2 for both:", get_r2(X, Y))

# built in scipy method produces the same answer for simple linear regression
from scipy import stats
y = [132, 143, 153, 162, 154, 168, 137, 149, 159, 128, 166]
x = [52, 59, 67, 73, 64, 74, 54, 61, 65, 46, 72]
x2 = [173, 184, 194, 211, 196, 220, 188, 188, 207, 167, 217]

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print("r2 for x2 only:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(x2,y)
print("r2 for x3 only:", r_value**2)

# multiple linear regression
from sklearn import linear_model
texts = np.array([y,x,x2]).T
clf = linear_model.LinearRegression()
clf.fit([t for t in texts],
    [t for t in texts])
# print(clf.coef_)

# from sklearn import linear_model
#
# reg = linear_model.LinearRegression()
# reg.fit(df[['B', 'C']], df['A'])
#
# >>> reg.coef_
# array([  4.01182386e-01,   3.51587361e-04])

# Other way to do it from stackoverflow: https://stackoverflow.com/questions/11479064/multiple-linear-regression-in-python
import statsmodels.api as sm

y = y
x = [x,x2]

def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

print("r2 for both:", reg_m(y, x).summary())
