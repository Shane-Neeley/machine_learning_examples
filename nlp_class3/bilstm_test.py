# https://deeplearningcourses.com/c/deep-learning-advanced-nlp
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU


T = 8 # e.g. number of words (sequence length)
D = 2 # e.g. dimension of word vectors (input dimensionality)
M = 3 # e.g. number of hidden units. (latent dimensionality) “units” is also called latent dimension

X = np.random.randn(1, T, D)
print("X:", X)
print("X.shape", X.shape)

input_ = Input(shape=(T, D))
print("input_", input_)
print("input_.shape", input_.shape)

# Bidirectional does forward and backward. Concatenates the inputs so we get a sequence of 2T.
# rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=True))
rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=False))

x = rnn(input_)

model = Model(inputs=input_, outputs=x)
model.summary()

# 5 things returned
# o = output
# h1, c1 = hidden state and cell state from FORWARD lstm
# h2, c2 = hidden state and cell state from BACKWARD lstm
o, h1, c1, h2, c2 = model.predict(X)
print("o:", o)
print("o.shape:", o.shape)
print("h1:", h1)
print("c1:", c1)
print("h2:", h2)
print("c2:", c2)
