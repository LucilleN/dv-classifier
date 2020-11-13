# credit: http://nicschrading.com/data/

# collects data from the publicly released data file
import json
from twython import Twython

# enter your APP_KEY and ACCESS_TOKEN from your Twitter API account here
twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

class Tweet():
    # A container class for tweet information
    def __init__(self, json, text, label, startIdx, endIdx, idStr):
        self.json = json
        self.text = text
        self.label = label
        self.id = idStr
        self.startIdx = startIdx
        self.endIdx = endIdx

    def __str__(self):
        return "id: " + self.id + " " + self.label + ": " + self.text

def collectTwitterData(twitter):
    tweetDict = {}
    # open the shared file and extract its data for all tweet instances
    with open("stayedLeftData.json") as f:
        for line in f:
            data = json.loads(line)
            label = data['label']
            startIdx = data['startIdx']
            endIdx = data['endIdx']
            idStr = data['id']
            tweet = Tweet(None, None, label, startIdx, endIdx, idStr)
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
            for tweetJSON in results:
                idStr = tweetJSON['id_str']
                tweet = tweetDict[idStr]
                tweet.json = tweetJSON
                # If this tweet was split, get the right part of the text
                if tweet.startIdx is not None:
                    tweet.text = tweetJSON['text'][tweet.startIdx : tweet.endIdx]
                # Otherwise get all the text
                else:
                    tweet.text = tweetJSON['text']
            i = 0
            chunk = []
    # get the rest (< 100 tweets)
    print("dumping rest...")
    results = twitter.lookup_status(id=chunk)
    for tweetJSON in results:
        idStr = tweetJSON['id_str']
        tweet = tweetDict[idStr]
        tweet.json = tweetJSON
        if tweet.startIdx is not None:
            tweet.text = tweetJSON['text'][tweet.startIdx : tweet.endIdx]
        else:
            tweet.text = tweetJSON['text']

    # return the Tweet objects in a list
    return list(tweetDict.values())