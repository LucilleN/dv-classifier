import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import trange

from data_loader import load_data
from utils import (build_vocab, eval_on_test_set, make_minibatch, parser,
                   strings_to_tensors)

PATH = 'models/gru.pt'


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


def train_model(use_og_data_only, n_epochs, bs, learning_rate):
    # If there's an available GPU, lets train on it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    Load the training data
    """

    posts_train, labels_train = load_data(
        og_file_path='data/train_reddit_submissions.csv',
        aug_file_path='data/train_synonym_augmented_reddit_submissions.csv',
        include_og=True,
        include_aug=not use_og_data_only)

    # need this since we're no longer using train_test_split
    np.random.shuffle(posts_train)
    np.random.shuffle(labels_train)

    tok_to_ix = build_vocab(posts_train)

    # Convert all posts to tensors that we will input into the model so that we do
    # not have to convert them again at every epoch
    train_data = strings_to_tensors(posts_train, tok_to_ix)

    labels_train = torch.LongTensor(labels_train)

    """
    Specify model's hyperparameters and architecture
    """

    hidden_size = 3
    num_layers = 1
    vocab_size = len(tok_to_ix)
    output_size = len(np.unique(labels_train))

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
                indices, train_data, labels_train)
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
            f'Epoch {epoch} / {n_epochs} | Loss: {running_loss / num_batches}')

    return model


if __name__ == "__main__":
    args = parser.parse_args()

    use_og_data_only = args.use_og_data_only
    bs = args.bs
    if args.retrain:
        n_epochs = args.n_epochs
        learning_rate = args.lr

        model = train_model(
            use_og_data_only, n_epochs, bs, learning_rate)
    else:
        model = torch.load(PATH)

    eval_on_test_set(model=model, use_og_data_only=use_og_data_only, bs=bs)
