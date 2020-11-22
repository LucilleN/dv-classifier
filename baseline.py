from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from data_loader import load_data

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