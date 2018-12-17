# Kit (Hong-Long Nguyen)
# This code was made possible thanks to linanqiu's template and tutorial.

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# perhaps k-means and DBSCAN here
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# numpy & random
import numpy
import random

# command line log
import logging
import sys

# csv reader
import csv

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

log.info('Model Save')
model = Doc2Vec.load('/Users/kit/OneDrive/Documents/GitHub/Tweet-Clustering/Clustering/tweets.d2v')

log.info('Partisanship')
train_arrays = numpy.zeros((15000, 100))
train_labels = numpy.zeros(15000)

for i in range(7500):
    prefix_train_pos = 'TRAIN_DEM_' + str(i)
    prefix_train_neg = 'TRAIN_REP_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[7500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[7500 + i] = 0

log.info(train_labels)

constituency = []
tweets = []

with open('test5-unsup.txt', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        constituency.append(row[0])
        tweets.append(row[1])

def process_tweet(tweet):
    return ''.join([x if x.isalnum() or x.isspace() else " " for x in tweet ]).split()

test_arrays = numpy.zeros((15000, 100))
test_labels = 15000 * [0]

for i in range(15000):
    test_arrays[i] = model.infer_vector(process_tweet(tweets[i]))
    if constituency[i] == "d":
        test_labels[i] = 1

log.info('Fitting')
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(train_arrays, train_labels)

#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          #intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

log.info(classifier.score(test_arrays, test_labels))
print "Accuracy:", (100 * classifier.score(test_arrays, test_labels)), "percent"
