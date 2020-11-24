import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data_loader import load_data, LABEL_TO_IX, IX_TO_LABEL

"""
Current Issues:
- one hot vectors vs tok_to_ix for translating sents to vectors
- word embeddings on RNN???
- batching during training?
"""

class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        input_size = vocab_size + hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.i2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data, prev_hidden):
        input = torch.cat((data, prev_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return hidden, output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def load_vocab(labelled_tweet):
    word_to_ix = {}
    for sent, _ in labelled_tweet:
        for token in sent:
            word_to_ix.setdefault(token, len(word_to_ix))
    return word_to_ix


def train_rnn(model, train, num_epochs, loss_fn, optimizer, tok_to_ix=None):
    enc = OneHotEncoder(sparse=False)
    print('training...')
    # losses, train_acc, valid_acc = [], [], []
    for epoch in range(num_epochs):
        model.train()
        hidden = rnn.initHidden()
        for post, label in train:
            x_train_tensor = torch.LongTensor(post.toarray()).view()
            pred, hidden = model(x_train_tensor, hidden)
            label_tensor = torch.Tensor([label])
            loss = loss_fn(pred, label_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("\nEpoch:", epoch)
        print("Training loss:", loss.item())


if __name__ == "__main__":
    posts, labels = load_data("data/reddit_submissions.csv")
    posts_train, posts_test, labels_train, labels_test = train_test_split(posts, labels, test_size=0.2)
    posts_labels_train = list(zip(posts_train, labels_train))

    vectorizer = CountVectorizer()
    train = zip(vectorizer.fit_transform(posts_train), posts_labels_train)
    vocab = vectorizer.vocabulary_

    # tok_to_ix = load_vocab(posts_labels_train)

    # hyperparameters
    learning_rate = .001
    n_epochs = 5

    rnn = RNN(vocab_size=len(vocab), hidden_size=128, output_size=3)

    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_rnn(model=rnn, train=train, loss_fn=criterion, optimizer=optimizer, num_epochs=n_epochs)
