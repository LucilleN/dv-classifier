"""
This script holds all our utility functions for common tasks within our project including:
- creating and manging minibatches
- loading and preprocessing data
- parsing command line arguments
- training / evaluation neural network models
"""
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import trange
from argparse import ArgumentParser

from data_loader import load_data


def parse_command_line_args():
    """
    Handles the parsing of command line arguments for all files.
    Returns `args`, the object that contains all the command line arguments,
    which can be accessed like this:
        args.retrain
        args.n_epochs
        ...etc.
    """
    parser = ArgumentParser(description='Neural DV-Classifier')

    parser.add_argument('--train_from_scratch', action='store_true', dest='retrain', default=False,
                        help='If set, trains the specified model from scratch. Any or all of the following flags can be set; any that are not set simply default to the existing default hyperparameters. If unset, will load the model in `models/`')
    parser.add_argument('--use_og_data_only', action='store_true', default=False,
                        help='If set, trains the model on only the original data rather than including augmented data.')
    parser.add_argument('--use_2_classes', action='store_true', default=False,
                        help='If set, trains with only 2 classes rather than the default 3: `0` for `DV-related` (combining critical and noncritical) and `1` for `general/unrelated`.')
    parser.add_argument('--n_epochs', action="store", type=int, default=10,
                        help='Takes in an integer; specifies the number of epochs to train for. Defaults to 10 epochs for all models.')
    parser.add_argument('--batch_size', action="store", type=int, default=10,
                        help='Takes in an integer; specifies the number of samples to use in each batch. Defaults to 50 samples for all models.')
    parser.add_argument('--learning_rate', action="store", type=float, default=0.01,
                        help='Takes in a float; specifies the learning rate to train with.')
    parser.add_argument('--hidden_size', action="store", type=int, default=25,
                        help='Takes in an int; specifies the hidden size dimension for the given NN to train on.')
    parser.add_argument('--num_layers', action="store", type=int, default=-1,
                        help='Takes in an int; specifies the number of layers within the specialized NN layer.')

    args = parser.parse_args()
    return args


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
        x_train_tensor = torch.LongTensor(x)
        tensors.append(x_train_tensor)
    return tensors


def reorder_minibatch(minibatch_data, minibatch_label):
    """
    Helper method that sorts the minibatch data and labels so that the samples are 
    sorted in order from longest to shortest. This is makes it more efficient to
    pad different-length samples, according to 
    https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch

    (Taken from Dr. Laurent's batching example)

    Returns a tuple of sorted lists: (data, labels)
    """
    temp = sorted(zip(minibatch_data, minibatch_label),
                  key=lambda x: len(x[0]), reverse=True)
    list1, list2 = map(list, zip(*temp))
    return list1, list2


def make_minibatch(indices, data, label):
    """
    Extracts a minibatch from the given data and label tensors using the given indices. 

    (Taken from Dr. Laurent's batching example)

    Returns a tuple of two tensors: (data, labels)
    """
    minibatch_data = [data[idx.item()] for idx in indices]
    minibatch_label = [label[idx.item()] for idx in indices]

    minibatch_data, minibatch_label = reorder_minibatch(
        minibatch_data, minibatch_label)
    minibatch_data = nn.utils.rnn.pad_sequence(minibatch_data, padding_value=1)
    minibatch_label = torch.stack(minibatch_label, dim=0)

    return minibatch_data, minibatch_label


def load_data_tensors(use_og_data_only):
    """
    Load the training data, converts them to tensors, and generates the
    vocabulary
    """
    posts_train, labels_train = load_data(
        og_file_path='data/train_reddit_submissions.csv',
        aug_file_path='data/train_synonym_augmented_reddit_submissions.csv',
        include_og=True,
        include_aug=not use_og_data_only)

    tok_to_ix = build_vocab(posts_train)

    """
    Convert all posts to tensors that we will input into the model so that we do
    not have to convert them again at every epoch
    """
    data_train = strings_to_tensors(posts_train, tok_to_ix)
    labels_train = torch.LongTensor(labels_train)

    return data_train, labels_train, tok_to_ix


def train_model(
        model,
        loss_func,
        optimizer,
        data_train,
        labels_train,
        n_epochs,
        batch_size,
        save_path,
        device):
    """
    Train the given model with the specified hyperparameters for n_epochs 
    and save the model to a file. 
    """
    epochs = trange(1, n_epochs + 1)
    for epoch in epochs:
        model.train()

        # Accumulate loss for all samples in this epoch
        running_loss = 0
        num_batches = 0
        num_samples = len(data_train)

        shuffled_indices = torch.randperm(num_samples)

        # Gradient descent algorithm for each data sample
        batches = trange(0, num_samples - batch_size, batch_size, leave=False)
        for count in batches:
            # Extract minibatches and send to GPU
            indices = shuffled_indices[count: count + batch_size]
            minibatch_data, minibatch_label = make_minibatch(
                indices, data_train, labels_train)
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

    torch.save(model, save_path)
    return model


def evaluate_model(model, tok_to_ix, device, use_og_data_only=True, bs=50):
    """ 
    Evaluates the given model on the specified test set.
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data, test_labels = load_data(
        og_file_path='data/test_reddit_submissions.csv',
        aug_file_path='data/test_synonym_augmented_reddit_submissions.csv',
        include_og=True,
        include_aug=not use_og_data_only)

    test_data = strings_to_tensors(test_data, tok_to_ix)
    test_labels = torch.LongTensor(test_labels)

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
