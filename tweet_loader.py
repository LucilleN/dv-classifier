"""
Taken from http://nicschrading.com/data/
A script that uses the Twitter API to get tweets given their tweet IDs
"""
# collects data from the publicly released data file
import json
import pandas as pd

from dotenv import load_dotenv
from twython import Twython
from os import getenv

load_dotenv()
# enter your APP_KEY and ACCESS_TOKEN from your Twitter API account here
twitter = Twython(app_key=getenv('TW_API_KEY'), app_secret=getenv(
    'TW_SECRET_KEY'), access_token=getenv('TW_BEARER_TOKEN'), token_type='bearer')


def requestTweet(row):
    try:
        tweet = twitter.show_status(id=row['post_id'])
        return tweet['text']
    except:
        print('could not find tweet:', row['post_id'])
        return ''


def collectTwitterData(twitter):
    # open the shared file and extract its data for all tweet instances
    data = pd.read_csv('data/dv_dataset_consolidated.csv')
    data['text'] = data.apply(lambda row: requestTweet(row), axis=1)
    print(data)


data = collectTwitterData(twitter)
# for tweet in data:
#     print(tweet)
