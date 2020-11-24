import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from progressbar import progressbar
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from data_loader import IX_TO_LABEL, LABEL_TO_IX, load_data

# If there's an available GPU, lets train on it
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

posts, labels = load_data('data/reddit_submissions.csv')
train_data, test_data, train_label, test_label = train_test_split(
    posts, labels, test_size=0.2)


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.rec = nn.LSTM(hidden_size, hidden_size,
                           num_layers=num_layers, bidirectional=True)

        # (hidden_size * 2) since bidirectional
        self.lin = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        # embeds = self.emb(input_seq)
        embeds = self.emb(input_seq)
        output_seq, (h_last, c_last) = self.rec(embeds.view(len(input_seq), 1, -1))

        h_direc_1 = h_last[0, :, :]
        h_direc_2 = h_last[1, :, :]
        h = torch.cat((h_direc_1, h_direc_2), dim=1)

        scores = self.lin(h)
        # return scores
        return self.sigmoid(scores)

def build_vocab(posts):
    tok_to_ix = {}
    for post in posts:
        tokens = post.split(' ')
        for token in tokens:
            tok_to_ix.setdefault(token, len(tok_to_ix))
    # Manually add our own placeholder tokens
    tok_to_ix.setdefault('<UNK>', len(tok_to_ix))
    tok_to_ix.setdefault('<s>', len(tok_to_ix))
    tok_to_ix.setdefault('<e>', len(tok_to_ix))
    return tok_to_ix

# vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=0)
# tok_to_ix = vectorizer.vocabulary_
# train_data = vectorizer.fit_transform(train_data)
tok_to_ix = build_vocab(train_data)

# Hyperparamaters
hidden_size = 3
learning_rate = 0.01
num_layers = 1
# vocab_size = len(vectorizer.vocabulary_)
vocab_size = len(tok_to_ix)
output_size = len(np.unique(train_label))
n_epochs = 3

model = LSTM(vocab_size=vocab_size,
             hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
# model = model.to(device)
# criterion = nn.NLLLoss()
criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    model.train()

    print("EPOCH {} OF {}".format(epoch, n_epochs))
    # Training performance tracking
    running_loss = 0

    counter = 0
    # for post, correct_label in progressbar(zip(train_data, train_label)):
    for post, correct_label in zip(train_data, train_label):
        counter += 1
        if counter % 200 == 0:
            print("sample {}".format(counter))
        tokens = post.split(' ')
        x = [tok_to_ix[tok] for tok in tokens if tok in tok_to_ix ]
        x_train_tensor = torch.LongTensor(x) #.to(device)

        # Feed net training data
        y_predicted_tensor = model(x_train_tensor)
        # print("predicted_scores: {}".format(y_predicted_tensor))

        y_true_list = np.zeros(output_size)
        y_true_list[correct_label] = 1.0
        y_true_tensor = torch.Tensor([y_true_list])
        # print("y_true_tensor: {}".format(y_true_tensor))

        # Backprop
        loss = criterion(y_predicted_tensor, y_true_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Compute metrics
        with torch.no_grad():
            running_loss += loss.item()

    print('t')
    print(f"""
    --------------
    Epoch #{epoch}
    Loss: {running_loss/len(train_label)}%
    --------------\n
    """)
