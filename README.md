# Comparing Partisanship Classification Approaches

This project uses the doc2vec approach to vectorize politician's tweets and then trains these on common supervised classifiers to categorize tweets into democratic vs republican. I also used unsupervised clustering too see how these clusters fare in splitting the tweets between partisan lines. As for the supervised classifiers, the ones tested are logistic regression, k  nearest neighbors,random forests, and support vector machine.

## Getting Started

### Prerequisites

You will need to install the following packages: nltk, numpy, gensim, and sklearn.

```
sudo pip install -U nltk
sudo pip install -U numpy
pip install --upgrade gensim
pip install -U scikit-learn
```

### Tweet Processing

The tweets already processed for a small doc2vec model are provided at [train8-dem.txt](train8-dem.txt), [train8-rep.txt](train8-rep.txt), [test4-dem.txt](test4-dem.txt), and [test4-rep.txt](test4-rep.txt).
The processed tweets to train a large doc2vec model are [train825-dem.txt](train825-dem.txt), [train825-rep.txt](train825-rep.txt), [test4-dem.txt](test4-dem.txt), and [test4-rep.txt](test4-rep.txt).

If you want replicate processing data to train on the small model yourself, however, the raw datasets can be found at [dataset8.csv](dataset8.csv) and [dataset4.csv](dataset4.csv). To process the raw data, execute:

```
python csv2partisan_train8.py
python csv2partisan_test4.py
```

### Tweet Vectorizing/Building the Doc2Vec Model

The code provided can build both a small and large doc2vec model as described in the paper, but in the interest of time, I highly recommend to only build the small model, which should only take a couple minutes (5-10).

```
# To build the small model, execute:
python vectorize.py

# If you really want to test the large model, run:
python vectorize.py -large
```

## Running the Experiments

To run all the supervised classification algorithms and see and compare their accuracies, execute:

```
# If you have built a small model,
python classify.py

# If you have a large model,
python classify.py -large
```


To run all the unsupervised clustering algorithms and compare their accuracies, run:
```
python cluster.py

# or...
python classify.py -large
```

## Author

**Kit (Hong-Long Nguyen)**

## Acknowledgments

* Zhi Xing's project on detecting Twitter sentiments that I based some of my scripts on: github.com/zxing01/deep-learning-twitter-sentiment
* Linan Qiu's tutorial on doc2vec that I closely followed: https://github.com/linanqiu/word2vec-sentiments
* The paper that introduced the doc2vec approach, referred to as parargraph vectors back then: https://arxiv.org/pdf/1405.4053.pdf
