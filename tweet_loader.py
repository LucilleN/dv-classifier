"""
Taken from http://nicschrading.com/data/
A script that uses the Twitter API to get tweets given their tweet IDs
"""
# collects data from the publicly released data file
import json
import pandas as pd
from twython import Twython
from config import API_KEY, SECRET_KEY, BEARER_TOKEN

# enter your APP_KEY and ACCESS_TOKEN from your Twitter API account here
twitter = Twython(app_key=API_KEY, app_secret=SECRET_KEY, access_token=BEARER_TOKEN, token_type='bearer')
print("twitter object: {}".format(twitter))

class Tweet():
    # A container class for tweet information
    def __init__(self, json, text, label, idStr):
        self.json = json
        self.text = text
        self.label = label
        self.id = idStr

    def __str__(self):
        return "id: {}, label: {}, text: {}".format(self.id, self.label, self.text)


def collectTwitterData(twitter):
    tweetDict = {}
    data = pd.read_csv('data/dv_dataset_consolidated.csv')
    # open the shared file and extract its data for all tweet instances
    for _, row in data.iterrows():
        label = row['class']
        idStr = row['post_id']
        tweet = Tweet(None, None, label, idStr)
        tweetDict[idStr] = tweet

    # download the tweets JSON to get the text and additional info
    i = 0
    chunk = []
    for tweetId in tweetDict:
        # gather up 100 ids and then call Twitter's API
        chunk.append(tweetId)
        i += 1
        if i >= 100:
            print("dumping 100...")
            # Make the API call
            results = twitter.lookup_status(id=chunk)
            print("results: {}".format(results))
            for tweetJSON in results:
                print("going through results")
                idStr = tweetJSON['id_str']
                tweet = tweetDict[idStr]
                tweet.json = tweetJSON
                # If this tweet was split, get the right part of the text
                if tweet.startIdx is not None:
                    tweet.text = tweetJSON['text'][tweet.startIdx: tweet.endIdx]
                # Otherwise get all the text
                else:
                    tweet.text = tweetJSON['text']
            i = 0
            chunk = []
    # get the rest (< 100 tweets)
    print("dumping rest...")
    results = twitter.lookup_status(id=chunk)
    for tweetJSON in results:
        print("going through results")
        idStr = tweetJSON['id_str']
        tweet = tweetDict[idStr]
        tweet.json = tweetJSON
        if tweet.startIdx is not None:
            tweet.text = tweetJSON['text'][tweet.startIdx: tweet.endIdx]
        else:
            tweet.text = tweetJSON['text']
        print(tweet)

    # return the Tweet objects in a list
    return list(tweetDict.values())

data = collectTwitterData(twitter)
# for tweet in data:
#     print(tweet)