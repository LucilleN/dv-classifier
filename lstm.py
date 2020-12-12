import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data_loader import load_data
from utils import build_vocab, strings_to_tensors, train_model, evaluate_model


class LSTM(nn.Module):
    """
    Implementation of LSTM neural network. Set the init parameter `bidirectional` to True 
    to use a bidirectional LSTM (a BLSTM).
    """

    def __init__(self, vocab_size, hidden_size, output_size, num_layers, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.rec = nn.LSTM(hidden_size, hidden_size,
                           num_layers=num_layers, bidirectional=bidirectional)
        # When bidirectional, the output of the LSTM will be twice as long, so we need to
        # multiply the next layer's dims by 2
        linear_size = hidden_size if not bidirectional else hidden_size * 2
        self.lin = nn.Linear(linear_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        embeds = self.emb(input_seq)
        output_seq, (h_last, c_last) = self.rec(embeds)

        h = None

        if self.bidirectional:
            h_direc_1 = h_last[1, :, :]
            h_direc_2 = h_last[2, :, :]
            h = torch.cat((h_direc_1, h_direc_2), dim=1)
        else:
            h = h_last[0, :, :]

        scores = self.lin(h)
        return self.sigmoid(scores)


if __name__ == "__main__":
    # If there's an available GPU, lets train on it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    Load the training, validation, and test data
    """
    posts, labels = load_data(
        og_file_path='data/reddit_submissions.csv',
        aug_file_path='data/synonym_augmented_reddit_submissions.csv',
        include_og=True,
        include_aug=True)

    posts_train, posts_test, train_labels, test_labels = train_test_split(
        posts, labels, test_size=0.2)

    tok_to_ix = build_vocab(posts_train)

    """
    Convert all posts to tensors that we will input into the model so that we do
    not have to convert them again at every epoch
    """
    train_data = strings_to_tensors(posts_train, tok_to_ix)
    test_data = strings_to_tensors(posts_test, tok_to_ix)

    train_labels = torch.LongTensor(train_labels)
    test_labels = torch.LongTensor(test_labels)

    """
    Specify model's hyperparameters and architecture
    """
    hidden_size = 25
    learning_rate = 0.01
    num_layers = 2
    vocab_size = len(tok_to_ix)
    output_size = len(np.unique(train_labels))
    n_epochs = 10
    bs = 50

    model = LSTM(vocab_size=vocab_size,
                 hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, bidirectional=False)
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_model(model=model, n_epochs=n_epochs, train_data=train_data, train_labels=train_labels,
                bs=bs, device=device, loss_func=loss_func, optimizer=optimizer)

    evaluate_model(model=model, test_data=test_data, test_labels=test_labels,
                   bs=bs, device=device)

"""
Output:

==============================

Evaluation:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       253
           1       0.66      0.71      0.69       324
           2       0.96      0.99      0.97      6465

    accuracy                           0.94      7042
   macro avg       0.54      0.57      0.55      7042
weighted avg       0.91      0.94      0.93      7042

==============================
"""
