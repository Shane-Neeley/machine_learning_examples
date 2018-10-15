# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python

# Author: http://lazyprogrammer.me
# from __future__ import print_function, division
# from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


# WHERE TO GET THE VECTORS:
# GloVe: https://nlp.stanford.edu/projects/glove/
# Direct link: http://nlp.stanford.edu/data/glove.6B.zip

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from util import y2indicator

def dist1(a, b):
    return np.linalg.norm(a - b)
def dist2(a, b):
    return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

# pick a distance type
dist, metric = dist2, 'cosine'
# dist, metric = dist1, 'euclidean'

###########################################

def find_analogies(w1, w2, w3):
  for w in (w1, w2, w3):
    if w not in word2vec:
      print("%s not in dictionary" % w)
      return

  king = word2vec[w1]
  man = word2vec[w2]
  woman = word2vec[w3]
  v0 = king - man + woman

  distances = pairwise_distances(v0.reshape(1, D), embedding, metric=metric).reshape(V)
  idxs = distances.argsort()[:4]
  for idx in idxs:
    word = idx2word[idx]
    if word not in (w1, w2, w3):
      best_word = word
      break

  print(w1, "-", w2, "=", best_word, "-", w3)

###########################################

def nearest_neighbors(w, n=10):
  if w not in word2vec:
    print("%s not in dictionary:" % w)
    return

  v = word2vec[w]
  distances = pairwise_distances(v.reshape(1, D), embedding, metric=metric).reshape(V)
  idxs = distances.argsort()[1:n+1]
  print("neighbors of: %s" % w)
  for idx in idxs:
    print("\t%s" % idx2word[idx])

###########################################

def load_vectors():
    # load in pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}
    embedding = []
    idx2word = []
    d = str(50)
    print('dimensions: ' + d)
    with open('../large_files/glove.6B/glove.6B.' + d + 'd.txt', encoding='utf-8') as f:
      # is just a space-separated text file in the format:
      # word vec[0] vec[1] vec[2] ...
      for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
        embedding.append(vec)
        idx2word.append(word)
    print('Found %s word vectors.' % len(word2vec))
    embedding = np.array(embedding)
    V, D = embedding.shape
    return word2vec, embedding, idx2word, d

###########################################

def get_data():
    print('loading data ...')
    data_train = []
    targets_train = []
    data_test = []
    targets_test = []
    with open('../large_files/r8-train-all-terms.txt', encoding='utf-8') as f1:
        for line in f1:
            values = line.split('\t')
            data_train.append(values[1])
            targets_train.append(values[0])
    with open('../large_files/r8-test-all-terms.txt', encoding='utf-8') as f2:
        for line in f2:
            values = line.split('\t')
            data_test.append(values[1])
            targets_test.append(values[0])

    # one-hot encode targets
    # how do i know it's assigning the same labels to each?
    Ytrain_labels = LabelEncoder().fit_transform(targets_train)
    Ytest_labels = LabelEncoder().fit_transform(targets_test)
    Ytrain = y2indicator(Ytrain_labels)
    Ytest = y2indicator(Ytest_labels)
    print('Ytrain: ', Ytrain.shape)
    print('Ytest: ', Ytest.shape)
    # possible shape problem if K test != K train
    if (Ytrain.shape[1] != Ytest.shape[1]):
        raise ValueError('A very specific bad thing happened.')

    # get an average word vector for the data
    def avgwords(data):
        tot = []
        for article in data:
            totalwordvecs = []
            for word in article.split():
                if word in word2vec:
                    wvec = word2vec[word]
                    totalwordvecs.append(wvec)
                else:
                    # if word not vectorized, return all zeros
                    totalwordvecs.append(np.zeros(int(d)))
            totalwordvecs = np.array(totalwordvecs)
            avgword = np.mean(totalwordvecs, axis=0)
            tot.append(avgword.tolist())
        return np.array(tot)

    Xtrain = avgwords(data_train)
    Xtest = avgwords(data_test)
    print('Xtrain: ', Xtrain.shape)
    print('Xtest: ', Xtest.shape)
    return Xtrain, Xtest, Ytrain, Ytest, Ytrain_labels, Ytest_labels

##########################################

def runkeras():

    from keras.models import Sequential
    from keras.layers import Dense, Activation
    import matplotlib.pyplot as plt

    X = np.concatenate((Xtrain, Xtest), axis=0)
    Y = np.concatenate((Ytrain, Ytest), axis=0)
    print(Y)
    N, D = X.shape
    K = Y.shape[1]

    # the model will be a sequence of layers
    model = Sequential()

    # ANN with layers [29 (D)] -> [500] -> [300] -> [2]
    model.add(Dense(units=500, input_dim=D))
    model.add(Activation('relu'))
    model.add(Dense(units=300)) # don't need to specify input_dim
    model.add(Activation('relu'))
    model.add(Dense(units=K))
    model.add(Activation('softmax'))

    # list of losses: https://keras.io/losses/
    # list of optimizers: https://keras.io/optimizers/
    # list of metrics: https://keras.io/metrics/
    model.compile(
      loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
    )

    # note: multiple ways to choose a backend
    # either theano, tensorflow, or cntk
    # https://keras.io/backend/
    # gives us back a <keras.callbacks.History object at 0x112e61a90>
    r = model.fit(X, Y, validation_split=0.25, epochs=60, batch_size=50)
    print("Returned:", r)

    # print the available keys
    # should see: dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])
    print(r.history.keys())

    # plot some data
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # accuracies
    plt.plot(r.history['acc'], label='acc')
    plt.plot(r.history['val_acc'], label='val_acc')
    plt.legend()
    plt.show()

###########################################

def logisticregression():
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics
    from sklearn.cross_validation import cross_val_score
    # instantiate a logistic regression model, and fit with X and y
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain[:,0:1])
    print("Train accuracy:", model.score(Xtrain, Ytrain[:,0:1])) # can't do multiclass
    print("Test accuracy:", model.score(Xtest, Ytest[:,0:1]))

    # evaluate the model by splitting into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state=0)
    # model2 = LogisticRegression()
    # model2.fit(X_train, y_train)
    # # predict class labels for the test set
    # predicted = model2.predict(X_test)
    # print predicted
    # # generate class probabilities
    # probs = model2.predict_proba(X_test)
    # print probs

###########################################

def nb():
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB()
    clf.fit(Xtrain, Ytrain_labels)
    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    print(clf.predict(Xtrain[2:300]))
    print(clf.score(Xtest, Ytest_labels))

if __name__ == '__main__':
    word2vec, embedding, idx2word, d = load_vectors()
    Xtrain, Xtest, Ytrain, Ytest, Ytrain_labels, Ytest_labels = get_data()
    runkeras()
    # logisticregression()
    #nb()




#hey
