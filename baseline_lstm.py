import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from data_loader import IX_TO_LABEL, LABEL_TO_IX, load_data


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
        output_seq, (h_last, c_last) = self.rec(
            embeds.view(len(input_seq), 1, -1))

        h = None

        if self.bidirectional:
            h_direc_1 = h_last[1, :, :]
            h_direc_2 = h_last[2, :, :]
            h = torch.cat((h_direc_1, h_direc_2), dim=1)
        else:
            h = h_last[0, :, :]

        scores = self.lin(h)
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
        x = [tok_to_ix[tok] if tok in tok_to_ix else tok_to_ix['<UNK>']
             for tok in tokens]
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
    hidden_size = 25
    learning_rate = 0.01
    num_layers = 2
    vocab_size = len(tok_to_ix)
    output_size = len(np.unique(train_labels))
    n_epochs = 10

    model = LSTM(vocab_size=vocab_size,
                 hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, bidirectional=True)
    model = model.to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    """
    Train the model
    """

    for epoch in range(1, n_epochs + 1):
        model.train()

        print(
            f"""\n--------------\nBeginning Epoch {epoch} of {n_epochs}\n--------------\n""")

        # Accumulate loss for all samples in this epoch
        running_loss = 0

        num_samples = len(train_data)
        sample_counter = 0

        # Gradient descent algorithm for each data sample
        for x_train_tensor, correct_label in zip(train_data, train_labels):

            sample_counter += 1
            if sample_counter % 1000 == 0:
                print("  > sample {} of {}".format(
                    sample_counter, num_samples))

            # Make a prediction on the training data
            y_predicted_tensor = model(x_train_tensor)

            # Create the true label's tensor by creating a zeros-array with the element at the
            # correct label's index set to 1.0.
            y_true_list = np.zeros(output_size)
            y_true_list[correct_label] = 1.0
            y_true_tensor = torch.Tensor([y_true_list]).to(device)

            # Backpropagation
            loss = loss_func(y_predicted_tensor, y_true_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute metrics
            with torch.no_grad():
                running_loss += loss.item()

        print(
            f"""\n--------------\nEnd of Epoch {epoch}\nLoss: {running_loss/sample_counter}\n--------------\n""")

    """ 
    Evaluate the model
    """
    with torch.no_grad():
        model.eval()
        predicted_labels = []
        for x_test_tensor, correct_label in zip(test_data, test_labels):
            y_predicted_tensor = model(x_test_tensor)
            y_predicted_tensor = y_predicted_tensor.to('cpu')

            # Store the labels that the model predicts so that we can calculate the accuracy, etc. later
            predicted_label = np.argmax(y_predicted_tensor.data.numpy())
            predicted_labels.append(predicted_label)

        print("\n==============================")
        print("\nEvaluation")
        report = classification_report(
            y_true=test_labels, y_pred=predicted_labels)
        print(report)
        print("\n==============================")


"""
Output:

==============================

Evaluation
              precision    recall  f1-score   support

           0       1.00      0.03      0.05       260
           1       0.70      0.68      0.69       335
           2       0.95      0.99      0.97      6447

    accuracy                           0.94      7042
   macro avg       0.89      0.57      0.57      7042
weighted avg       0.94      0.94      0.93      7042

==============================
"""
