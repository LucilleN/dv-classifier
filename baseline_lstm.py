import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import trange

from data_loader import load_data
from utils import build_vocab, make_minibatch, strings_to_tensors


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
        aug_file_path='data/augmented_reddit_submissions.csv', 
        include_og=True, 
        include_aug=True,
        fraction_class_2_to_load=0.1)

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
                 hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, bidirectional=True)
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
            # Extract minibatches and send to GPU
            indices = shuffled_indices[count : count + bs]
            minibatch_data, minibatch_label = make_minibatch(
                indices, train_data, train_labels)
            minibatch_data, minibatch_label = minibatch_data.to(
                device), minibatch_label.to(device)

            # Make a predictions on the training data
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
            f'Epoch {epoch}/{n_epochs} | Loss: {running_loss/num_batches}')

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
