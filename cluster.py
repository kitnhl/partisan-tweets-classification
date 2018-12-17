# Kit (Hong-Long Nguyen)

import numpy, sys
from gensim.models import Doc2Vec
from sklearn.cluster import KMeans, AgglomerativeClustering

NUM_CLUSTERS = 2

print "Loading Model..."
# check whether clustering using the large doc2vec model is desired
try:
    if sys.argv[1] == '-large':
        model = Doc2Vec.load('largemodel.d2v')
    else:
        raise IndexError
except IndexError:
    model = Doc2Vec.load('smallmodel.d2v')

print "Loading Test Vector Arrays..."
test_arrays = numpy.zeros((15000, 100))
test_labels = 15000 * [0]

for i in range(7500):
    prefix_test_pos = 'TEST_DEM_' + str(i)
    prefix_test_neg = 'TEST_REP_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[7500 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1


print "K Means running..."
kmeans = KMeans(NUM_CLUSTERS)
kmeans.fit(test_arrays)
km_labels = kmeans.labels_

correct_km_labels = sum([x ^ y for x, y in zip(km_labels, test_labels)])
if correct_km_labels < 7500:
	correct_km_labels = 15000 - correct_km_labels

km_accuracy = (correct_km_labels / 15000.) * 100
print "K Means accuracy:", '%.3f'%(km_accuracy), "%"


print "Agglomerative Clustering running..."
agglomerative = AgglomerativeClustering(NUM_CLUSTERS)
agglomerative.fit(test_arrays)
aggl_labels = agglomerative.labels_

correct_aggl_labels = sum([x ^ y for x, y in zip(aggl_labels, test_labels)])
if correct_aggl_labels < 7500:
	correct_aggl_labels = 15000 - correct_aggl_labels

aggl_accuracy = (correct_aggl_labels / 15000.) * 100
print "Agglomerative Clustering accuracy:", '%.3f'%(aggl_accuracy), "%"