import csv

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import numpy as np

# The domesticviolence and survivorsofabuse subreddits will be class 0, critical; these are personal stories, calls for help, requests for advice
# The abuseInterrupted subreddit will be class 1, noncritical; it mostly contains empty text, links to articles, general statements about abuse, etc.
# Everything else will be class 2, general/unrelated
CLASSES = {
    'relationship_advice': 2, # 5874 samples
    'relationships': 2, # 8201 samples
    'casualconversation': 2, # 7286 samples
    'advice': 2, # 5913 samples
    'anxiety': 2, # 4183 samples
    'anger': 2, # 837 samples
    'abuseinterrupted': 1, # 1653 samples
    'domesticviolence': 0, # 749 samples
    'survivorsofabuse': 0 # 512 samples
}

def load_data(file_path):
    """
    Loads text and labels from dataset stored in file_path, a CSV.
    """
    posts = []
    labels = []

    with open(file_path) as f:
        reader = csv.reader(f)
        # Skip the first row that just has column names
        rows = list(reader)[1:]
        for row in rows:
            # print("\n" + str(row))
            subreddit_name = row[0]
            label = CLASSES[subreddit_name]
            post_title = row[2]
            post_text = row[3]
            post_title_and_text = post_title + " " + post_text
            labels.append(label)
            posts.append(post_title_and_text)

    return (np.array(posts), np.array(labels))

if __name__ == "__main__":
    posts, labels = load_data("data/reddit_submissions.csv")
    posts_train, posts_test, labels_train, labels_test = train_test_split(posts, labels, test_size=0.2)
    
    print("posts_train: {}".format(posts_train.shape))
    print("labels_train: {}".format(labels_train.shape))

    """
    Train and evaluate on training dataset
    """
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    features_train = vectorizer.fit_transform(posts_train)
    
    model = LogisticRegression()
    model.fit(features_train, labels_train)

    score_train = model.score(features_train, labels_train)
    print('Training set score: ', score_train)

    """
    Test on held-out dataset
    """
    features_test = vectorizer.transform(posts_test)
    score_test = model.score(features_test, labels_test)
    print('Testing set score: ', score_test)