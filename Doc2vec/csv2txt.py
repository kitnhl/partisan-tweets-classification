from string import punctuation
import re
from nltk.tokenize import TweetTokenizer
import random

import csv

user_ids = []
tweets = []
partisanship = {}

with open('dataset8.csv', 'r') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        user_ids.append(row[1])
        tweets.append(row[3])

with open('pol_accounts.csv', 'r') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        if "democrat" in row[9].lower():
            partisanship[row[0]] = 'd'
        if "republic" in row[9].lower():
            partisanship[row[0]] = 'r'

# from sentiment140_clean.py
def write_to_file(file_name, sents):
    with open(file_name, 'w') as f:
        for i, sent in enumerate(sents):
            #print sent
            #f.write(sent.encode('utf-8'))
            '''
            try:
                f.write(sent)
            except UnicodeEncodeError:
                continue
            '''
            f.write(user_ids[i])
            f.write(', ')
            try:
                f.write(partisanship[user_ids[i]])
            except KeyError:
                f.write('u')
            f.write(', ')
            f.write(sent)
            f.write('\n')

def clean_tweet(tweet):
    tknzr = TweetTokenizer()
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet.lower())
    tweet = ' '.join(tweet.split())
    words = tknzr.tokenize(tweet)
    words = [''.join(c for c in s if c not in punctuation) for s in words]
    words = [s for s in words if s]
    sent = " ".join(words)
    return sent


def clean_data(content, output_file_name):

    # Please notice that only part of the tweets are used.
    '''
    if len(content) > 1000:
        random.shuffle(content)
        content = content[0:100000]
    '''
    tweet_sents = []
    for line in content:
        sent = clean_tweet(line)
        tweet_sents.append(sent)
        '''
        try:
            sent = clean_tweet(line)
            tweet_sents.append(sent)
        except UnicodeDecodeError:
            continue
        '''

    write_to_file(output_file_name, tweet_sents)


if __name__ == '__main__':
    clean_data(tweets, 'unlabeled8.txt')