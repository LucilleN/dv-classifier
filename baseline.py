from twython import Twython
from config import API_KEY, SECRET_KEY, BEARER_TOKEN
from tweet_loader import collectTwitterData

twitter = Twython(API_KEY, access_token=BEARER_TOKEN)

if __name__ == "__main__":
    twitter_data = collectTwitterData(twitter)
    for tweet in twitter_data:
        print(tweet)
    # print(twitter_data)