from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import numpy as np

def load_data(self, file_path):
    """
    Loads text and labels from dataset stored in file_path, a CSV.
    """
    posts = []
    labels = []

    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0])
            post = ' '.join(list(row[1:]))
            labels.append(label)
            posts.append(post)

    return (np.array(posts), np.array(labels))

if __name__ == "__main__":
    posts, labels = load_data()
    posts_train, posts_test, labels_train, labels_test = train_test_split(posts, labels, test_size=0.2)
    
    """
    Train and evaluate on training dataset
    """
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    features_train = vectorizer.fit_transform(posts)
    
    model = LogisticRegression()
    model.fit(features_train, labels_train)

    score_train = model.score(features_train, labels_train)
    print('Training set score: ', score_train)

    """
    Test on held-out dataset
    """
    features_test = vectorizer.transform(posts_test)
    score_test = model.score(features_test, labels_test)
    print('Training set score: ', score_test)