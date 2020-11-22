import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import numpy as np

from data_loader import load_data, LABEL_TO_IX, IX_TO_LABEL


class LSTM(nn.Module):
    """
    Implementation of LSTM neural network. Set the init parameter `bidirectional` to True 
    to use a bidirectional LSTM.
    """
    def __init__(self, num_words, emb_dim, num_y, hidden_dim=32, bidirectional=False):
        super().__init__()
        self.emb = nn.Embedding(num_words, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, 
                            bidirectional=bidirectional, batch_first=True)
        # When bidirectional, the output of the LSTM will be twice as long, so we need to 
        # multiply the next layer's dims by 2
        linear_dims = hidden_dim if not bidirectional else hidden_dim * 2
        self.linear = nn.Linear(linear_dims, num_y)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, text):
        embeds = self.emb(text)
        lstm_out, _ = self.lstm(embeds.view(len(text), 1, -1))
        # label_space = self.linear(lstm_out.view(len(text), -1))
        # return self.softmax(label_space)
        return self.linear(lstm_out)


def predict_for_one_post(model, post, tok_to_ix):
    """
    Makes a prediction with the given model on a single sentence (list of tokens) by
    converting the sentence into an input feature vector using the given tok_to_ix mapping. 
    Returns the tensor representing the NN model's prediction. 
    """
    post = post.split(' ')
    x = [tok_to_ix[tok] for tok in post if tok in tok_to_ix ]
    x_tensor = torch.LongTensor(x)
    return model(x_tensor)


def get_accuracy(true_labels, predicted_labels):
    """
    Given a list of true labels and a list of predicted labels, returns the 
    accuracy of the predicted labels as a decimal value between 0 and 1. 
    """
    # Assign a score of 1 for each correct prediction and 0 for each incorrect prediction
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    correct_predictions = np.where(true_labels == predicted_labels, 1, 0)
    return np.mean(correct_predictions)


def pred_tensor_to_label(pred_y_tensor, ix_to_label):
    """
    Converts a prediction tensor from the output of a NN model to a list of predicted tags.
    The given pred_y_tensor produced by the NN model contains inner lists for each word
    in the sentence, and each inner list contains confidence values for all the options of
    semantic tags for that word. This method finds the indices with the highest confidence
    for each word and builds a list of the tags that correspond to those indices.
    """
    most_confident_index = np.argmax(pred_y_tensor.data.numpy())
    predicted_label = ix_to_label[most_confident_index]
    return predicted_label


def train_nn(model, optimizer, loss_fn, n_epochs, posts_train, labels_train, tok_to_ix, label_to_ix, ix_to_label):
    """
    Trains the given model with the given optimizer and loss function on the post and label training sets.
    """
    accuracy = -1.0
    for epoch in range(n_epochs):
        model.train()
        predicted_labels = []
        for post, correct_label in zip(posts_train, labels_train):
            # Make a prediction with the input training sentence
            pred_y_tensor = predict_for_one_post(model, post, tok_to_ix)
            print("pred_y_tensor: {}".format(pred_y_tensor))
            pred_y_label = pred_tensor_to_label(pred_y_tensor, ix_to_label)
            predicted_labels.append(pred_y_label)

            # Adjust weights through backpropagation by comparing our predicted tags to the correct tags
            y = label_to_ix[correct_label]
            y_train_tensor = torch.LongTensor(y)

            loss = loss_fn(pred_y_tensor, y_train_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        accuracy = get_accuracy(labels_train, predicted_labels)
        print("  >  Epoch {}: Loss={:.5f}, Accuracy={:.5f}".format(epoch, loss.item(), accuracy))


if __name__ == "__main__":
    posts, labels = load_data("data/reddit_submissions.csv")
    posts_train, posts_test, labels_train, labels_test = train_test_split(posts, labels, test_size=0.2)
    
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    vectorizer.fit(posts_train)
    
    vocab = vectorizer.vocabulary_
    unique_labels = np.unique(labels_train)

    print("=================================")
    print("LSTM")
    print("=================================")

    # Hyperparameters
    emb_dim = 10
    learning_rate = 0.01
    n_epochs = 10

    model_LSTM = LSTM(num_words=len(vocab), emb_dim=emb_dim, num_y=len(unique_labels))
    optimizer = optim.Adam(model_LSTM.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    train_nn(model_LSTM, optimizer, loss_fn, n_epochs, posts_train, labels_train, tok_to_ix=vocab, label_to_ix=LABEL_TO_IX, ix_to_label=IX_TO_LABEL)
    