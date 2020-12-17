"""
This script trains and evaluates our baseline model, a Logistic Regression classifier.

To run this script, simply execute: `python3 baseline_lr.py`. You can also optionally set the
flag --use_og_data_only to train and test on only the original data without the augmented data.

Example Usage: 
    python3 baseline_lr.py --use_og_data_only
"""


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse

from data_loader import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_og_data_only',
                        action='store_true', help='If set, then omits augmented data from training and testing sets')
    args = parser.parse_args()

    posts_train, labels_train = load_data(
        og_file_path="data/train_reddit_submissions.csv", 
        aug_file_path="data/train_synonym_augmented_reddit_submissions.csv", 
        include_og=True, 
        include_aug=not args.use_og_data_only)
    posts_test, labels_test = load_data(
        og_file_path="data/test_reddit_submissions.csv", 
        aug_file_path="data/test_synonym_augmented_reddit_submissions.csv", 
        include_og=True, 
        include_aug=not args.use_og_data_only)

    vectorizer = CountVectorizer(ngram_range=(1, 1))
    features_train = vectorizer.fit_transform(posts_train)

    """
    Train on 80% training set
    """
    model_LR = LogisticRegression(max_iter=1000)
    print("Training...")
    model_LR.fit(features_train, labels_train)

    score_train = model_LR.score(features_train, labels_train)
    print('LR Training set score: ', score_train)

    """
    Test on 20% held-out dataset
    """
    features_test = vectorizer.transform(posts_test)
    score_test = model_LR.score(features_test, labels_test)
    print('LR Testing set score: ', score_test)

    predictions_test = model_LR.predict(features_test)
    report = classification_report(labels_test, predictions_test)
    print(report)