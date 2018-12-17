# Kit (Hong-Long Nguyen)
# This code was based off Zhi Xing's template.

from string import punctuation
import re
from nltk.tokenize import TweetTokenizer
import random

import csv

user_ids = []
tweets = []
partisanship = {}

with open('dataset4.csv', 'r') as f:
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

def write_to_dem_file(file_name, sents):
    with open(file_name, 'w') as f:
        for i, sent in enumerate(sents):
            try:
                party = partisanship[user_ids[i]]
            except KeyError:
                continue
            if party == 'd':    
                try:
                    f.write(sent)
                    f.write('\n')
                except UnicodeEncodeError:
                    continue

def write_to_rep_file(file_name, sents):
    with open(file_name, 'w') as f:
        for i, sent in enumerate(sents):
            try:
                party = partisanship[user_ids[i]]
            except KeyError:
                continue
            if party == 'r':    
                try:
                    f.write(sent)
                    f.write('\n')
                except UnicodeEncodeError:
                    continue

def clean_tweet(tweet):
    tknzr = TweetTokenizer()
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet.lower())
    tweet = ' '.join(tweet.split())
    words = tknzr.tokenize(tweet)
    words = [''.join(c for c in s if c not in punctuation) for s in words]
    words = [s for s in words if s]
    sent = " ".join(words)
    return sent


def clean_data(content, dem_outfile, rep_outfile):
    tweet_sents = []
    for line in content:
        sent = clean_tweet(line)
        tweet_sents.append(sent)
        
    write_to_dem_file(dem_outfile, tweet_sents)
    write_to_rep_file(rep_outfile, tweet_sents)


if __name__ == '__main__':
    clean_data(tweets, 'test4-dem.txt', 'test4-rep.txt')