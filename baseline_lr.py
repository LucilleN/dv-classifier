from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse

from data_loader import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_og_data',
                        action='store_true', help='If set, then includes original data in training and testing sets')
    parser.add_argument('--use_aug_data',
                        action='store_true', help='If set, then includes augmented data in training and testing sets')
    args = parser.parse_args()

    posts, labels = load_data(
        og_file_path="data/reddit_submissions.csv", 
        aug_file_path="data/synonym_augmented_reddit_submissions.csv", 
        include_og=args.use_og_data, 
        include_aug=args.use_aug_data)

    posts_train, posts_test, labels_train, labels_test = train_test_split(posts, labels, test_size=0.2)
    
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    features_train = vectorizer.fit_transform(posts_train)

    print("=================================")
    print("LOGISTIC REGRESSION")
    print("=================================")

    """
    Train on 80% training set
    """
    model_LR = LogisticRegression()
    model_LR.fit(features_train, labels_train)

    score_train = model_LR.score(features_train, labels_train)
    print('  LR Training set score: ', score_train)

    """
    Test on 20% held-out dataset
    """
    features_test = vectorizer.transform(posts_test)
    score_test = model_LR.score(features_test, labels_test)
    print('  LR Testing set score: ', score_test)

    predictions_test = model_LR.predict(features_test)
    report = classification_report(labels_test, predictions_test)
    print(report)