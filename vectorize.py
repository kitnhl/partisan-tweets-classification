# Kit (Hong-Long Nguyen)
# This code was made possible thanks to Linan Qiu's template and tutorial.

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy & random
import numpy
import random

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

# check whether a large doc2vec model is desired
large = False
try:
    if sys.argv[1] == '-large':
        large = True
except IndexError:
    pass


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled

log.info('Loading Source')
if large:
    sources = {'test4-dem.txt':'TEST_DEM', 'test4-rep.txt':'TEST_REP', 'train825-dem.txt':'TRAIN_DEM', 'train825-rep.txt':'TRAIN_REP'}
else:
    sources = {'test4-dem.txt':'TEST_DEM', 'test4-rep.txt':'TEST_REP', 'train8-dem.txt':'TRAIN_DEM', 'train8-rep.txt':'TRAIN_REP'}

log.info('TaggedDocument')
tweets = LabeledLineSentence(sources)

log.info('D2V')
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7, iter=10)
model.build_vocab(tweets.to_array())

log.info('Epoch')
model.train(tweets.sentences_perm(), epochs=model.iter, total_examples=model.corpus_count)

log.info('Saving Model')
if large:
    model.save('largemodel.d2v')
    log.info('Large Model Completed')
else:
    model.save('smallmodel.d2v')
    log.info('Small Model Completed')