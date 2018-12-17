# Kit (Hong-Long Nguyen)
# This code was made possible thanks to Linan Qiu's and Zhi Xing's templates and tutorials.

# gensim modules
from gensim.models import Doc2Vec

# classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# numpy
import numpy

# command line log
import logging
import sys

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

# check whether classification using the large doc2vec model is desired
large = False
try:
    if sys.argv[1] == '-large':
        large = True
except IndexError:
    pass

log.info('Loading Model')
if large:
    model = Doc2Vec.load('largemodel.d2v')
    n_arrays = 45000
else:
    model = Doc2Vec.load('smallmodel.d2v')
    n_arrays = 15000

log.info('Loading Vector Array')
train_arrays = numpy.zeros((n_arrays, 100))
train_labels = numpy.zeros(n_arrays)

for i in range(n_arrays / 2):
    prefix_train_pos = 'TRAIN_DEM_' + str(i)
    prefix_train_neg = 'TRAIN_REP_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[n_arrays / 2 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[n_arrays / 2 + i] = 0


test_arrays = numpy.zeros((15000, 100))
test_labels = numpy.zeros(15000)

for i in range(7500):
    prefix_test_pos = 'TEST_DEM_' + str(i)
    prefix_test_neg = 'TEST_REP_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[7500 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[7500 + i] = 0


log.info('Fitting')
print "Logistic Regression running..."
lr_classifier = LogisticRegression(solver='lbfgs')
lr_classifier.fit(train_arrays, train_labels)
print "Logistic Regresssion accuracy:", \
    '%.3f'%(100 * lr_classifier.score(test_arrays, test_labels)), "%"

print "K Nearest Neighbors running..."
knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(train_arrays, train_labels)
print "K Nearest Neighbors accuracy:", \
    '%.3f'%(100 * knn_classifier.score(test_arrays, test_labels)), "%"

print "Random Forests running..."
rf_classifier = RandomForestClassifier(n_estimators=25)
rf_classifier.fit(train_arrays, train_labels)
print "Random Forests accuracy:", \
    '%.3f'%(100 * rf_classifier.score(test_arrays, test_labels)), "%"

print "Support Vector Machine running..."
svc_classifier = SVC(gamma='scale')
svc_classifier.fit(train_arrays, train_labels)
print "Support Vector Machine accuracy:", \
    '%.3f'%(100 * svc_classifier.score(test_arrays, test_labels)), "%"
