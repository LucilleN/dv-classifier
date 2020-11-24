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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def forward(self, input_seq):
        embeds = self.emb(input_seq)
        output_seq, (h_last, c_last) = self.rec(embeds)

        h_direc_1 = h_last[2, :, :]
        h_direc_2 = h_last[3, :, :]
        h = torch.cat((h_direc_1, h_direc_2), dim=1)

        scores = self.lin(h)
        return scores


vectorizer = CountVectorizer(ngram_range=(1, 1))
train_data = vectorizer.fit_transform(train_data)

# Hyperparamaters
hidden_size = 10
learning_rate = 0.01
num_layers = 2
vocab_size = len(vectorizer.vocabulary_)
output_size = len(np.unique(train_label))
n_epochs = 10

model = LSTM(vocab_size=vocab_size,
             hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    model.train()

    # Training performance tracking
    running_loss = 0

    for post, correct_label in progressbar(zip(train_data, train_label)):
        x_train_tensor = torch.LongTensor(
            post.toarray()).view(vocab_size, 1).to(device)
        y_train_tensor = torch.LongTensor([correct_label]).to(device)

        # Feed net training data
        scores = model(x_train_tensor)

        # Backprop
        loss = criterion(scores, y_train_tensor)
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
