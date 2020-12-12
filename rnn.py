import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data_loader import load_data
from utils import build_vocab, strings_to_tensors, train_model, evaluate_model


class RNN(nn.Module):
    """
    Implementation of RNN neural network.
    """

    def __init__(self, vocab_size, hidden_size, output_size, num_layers, bidirectional=False):
        super().__init__()
        # self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size,
                          num_layers=num_layers, batch_first=False)
        linear_size = hidden_size if not bidirectional else hidden_size * 2
        self.lin = nn.Linear(linear_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        embeds = self.emb(input_seq)
        (out, hn_last) = self.rnn(embeds)

        out = out[0, :, :]

        scores = self.lin(out)
        return self.sigmoid(scores)


if __name__ == "__main__":
    # If there's an available GPU, lets train on it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    Load the training, validation, and test data
    """

    posts, labels = load_data('data/reddit_submissions.csv')

    posts_train, posts_test, train_labels, test_labels = train_test_split(
        posts, labels, test_size=0.2)

    tok_to_ix = build_vocab(posts_train)

    # Convert all posts to tensors that we will input into the model so that we do
    # not have to convert them again at every epoch
    train_data = strings_to_tensors(posts_train, tok_to_ix)
    test_data = strings_to_tensors(posts_test, tok_to_ix)

    train_labels = torch.LongTensor(train_labels)
    test_labels = torch.LongTensor(test_labels)

    """
    Specify model's hyperparameters and architecture
    """

    hidden_size = 3
    learning_rate = 0.01
    num_layers = 1
    vocab_size = len(tok_to_ix)
    output_size = len(np.unique(train_labels))
    n_epochs = 10
    bs = 50

    model = RNN(vocab_size=vocab_size,
                hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_model(model=model, n_epochs=n_epochs, train_data=train_data, train_labels=train_labels,
                bs=bs, device=device, loss_func=loss_func, optimizer=optimizer)

    evaluate_model(model=model, test_data=test_data, test_labels=test_labels,
                   bs=bs, device=device)
