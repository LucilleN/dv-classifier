import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import trange

from data_loader import load_data
from utils import build_vocab, make_minibatch, strings_to_tensors


class GRU(nn.Module):
    """
    Implementation of GRU neural network.
    """

    def __init__(self, vocab_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size,
                          num_layers=num_layers, batch_first=False)
        self.lin = nn.Linear(hidden_size, output_size)
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

    model = GRU(vocab_size=vocab_size,
                hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    """
    Train the model
    """
    epochs = trange(1, n_epochs + 1)
    for epoch in epochs:
        model.train()

        # Accumulate loss for all samples in this epoch
        running_loss = 0
        num_batches = 0
        num_samples = len(train_data)

        shuffled_indices = torch.randperm(num_samples)

        # Gradient descent algorithm for each data sample
        batches = trange(0, num_samples - bs, bs, leave=False)
        for count in batches:
            # Extract minibatches and send to device
            indices = shuffled_indices[count: count + bs]
            minibatch_data, minibatch_label = make_minibatch(
                indices, train_data, train_labels)
            minibatch_data, minibatch_label = minibatch_data.to(
                device), minibatch_label.to(device)

            # Make predictions on the training data
            scores = model(minibatch_data)

            # Backpropagation
            loss = loss_func(scores, minibatch_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute metrics
            with torch.no_grad():
                num_batches += 1
                running_loss += loss.item()

        epochs.set_description(
            f'Epoch {epoch} / {n_epochs} | Loss: {running_loss/num_batches}')

    """ 
    Evaluate the model
    """
    with torch.no_grad():
        model.eval()
        predicted_labels = []

        # cycle through test set in batches
        batches = trange(0, len(test_data) - bs, bs,
                         desc='evaluating on test set', leave=False)
        for i in batches:
            # extract minibatch
            indices = torch.arange(i, i + bs)
            minibatch_data, _ = make_minibatch(indices, test_data, test_labels)
            minibatch_data = minibatch_data.to(device)

            # make and score predictions
            scores = model(minibatch_data)
            predicted_labels.extend(scores.argmax(dim=1).tolist())

        # evaluate remaining samples
        indices = torch.arange(len(predicted_labels), len(test_labels))
        minibatch_data, _ = make_minibatch(indices, test_data, test_labels)
        minibatch_data = minibatch_data.to(device)

        scores = model(minibatch_data)
        predicted_labels.extend(scores.argmax(dim=1).tolist())

        print(classification_report(y_true=test_labels, y_pred=predicted_labels))
