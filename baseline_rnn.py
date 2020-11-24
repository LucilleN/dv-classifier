import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_loader import IX_TO_LABEL, LABEL_TO_IX, load_data


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
        (out, hn_last) = self.rnn(embeds.view(len(input_seq), 1, -1))

        out = out[0, :, :]

        scores = self.lin(out)
        return self.sigmoid(scores)


def build_vocab(posts):
    """
    Given the training set of posts, constructs the vocabulary dictionary, `tok_to_ix`, that
    maps unique tokens to their index in the vocabulary.
    """
    tok_to_ix = {}
    for post in posts:
        tokens = post.split(' ')
        for token in tokens:
            tok_to_ix.setdefault(token, len(tok_to_ix))
    # Manually add our own placeholder tokens
    tok_to_ix.setdefault('<UNK>', len(tok_to_ix))
    return tok_to_ix


def strings_to_tensors(posts_array, tok_to_ix):
    """
    Converts each string in an array of strings into a tensor that we will input into the model
    so that we don't have to convert each sample to a tensor again at each epoch.
    """
    tensors = []
    for post in posts_array:
        tokens = post.split(' ')
        x = [tok_to_ix[tok] if tok in tok_to_ix else tok_to_ix['<UNK>'] for tok in tokens]
        x_train_tensor = torch.LongTensor(x).to(device)
        tensors.append(x_train_tensor)
    return tensors


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

    """
    Specify model's hyperparameters and architecture
    """

    hidden_size = 3
    learning_rate = 0.01
    num_layers = 1
    vocab_size = len(tok_to_ix)
    output_size = len(np.unique(train_labels))
    n_epochs = 3

    model = RNN(vocab_size=vocab_size,
                hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
    model = model.to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    """
    Train the model
    """

    for epoch in range(1, n_epochs + 1):
        model.train()

        print(f"""\n--------------\nBeginning Epoch {epoch} of {n_epochs}\n--------------\n""")

        # Accumulate loss for all samples in this epoch
        running_loss = 0

        num_samples = len(train_data)
        sample_counter = 0

        # Gradient descent algorithm for each data sample
        for x_train_tensor, correct_label in zip(train_data, train_labels):

            sample_counter += 1
            if sample_counter % 1000 == 0:
                print("  > sample {} of {}".format(sample_counter, num_samples))

            # Make a prediction on the training data
            y_predicted_tensor = model(x_train_tensor)

            # Create the true label's tensor by creating a zeros-array with the element at the
            # correct label's index set to 1.0.
            y_true_list = np.zeros(output_size)
            y_true_list[correct_label] = 1.0
            y_true_tensor = torch.Tensor([y_true_list])

            # Backpropagation
            loss = loss_func(y_predicted_tensor, y_true_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute metrics
            with torch.no_grad():
                running_loss += loss.item()

        print(f"""\n--------------\nEnd of Epoch {epoch}\nLoss: {running_loss / len(train_labels)}\n--------------\n""")

    """ 
    Evaluate the model
    """
    with torch.no_grad():
        model.eval()
        predicted_labels = []
        for x_test_tensor, correct_label in zip(test_data, test_labels):
            y_predicted_tensor = model(x_test_tensor)

            # Store the labels that the model predicts so that we can calculate the accuracy, etc. later
            predicted_label = np.argmax(y_predicted_tensor.data.numpy())
            predicted_labels.append(predicted_label)

        print("\n==============================")
        print("\nEvaluation")
        report = classification_report(y_true=test_labels, y_pred=predicted_labels)
        print(report)
        print("\n==============================")
