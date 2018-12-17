# Kit (Hong-Long Nguyen)

import numpy, csv
from gensim.models import Doc2Vec
from sklearn.cluster import KMeans, AgglomerativeClustering

model = Doc2Vec.load('/Users/kit/OneDrive/Documents/GitHub/Tweet-Clustering/Clustering/smallmodel.d2v')

NUM_CLUSTERS = 2

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