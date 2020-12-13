import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import trange


    
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
    temp = sorted(zip(minibatch_data, minibatch_label),
                  key=lambda x: len(x[0]), reverse=True)
    list1, list2 = map(list, zip(*temp))
    return list1, list2


def make_minibatch(indices, data, label):
    """
    Extract minibatch from given data and label tensors using the given indices. 
    """
    minibatch_data = [data[idx.item()] for idx in indices]
    minibatch_label = [label[idx.item()] for idx in indices]

    minibatch_data, minibatch_label = reorder_minibatch(
        minibatch_data, minibatch_label)
    minibatch_data = nn.utils.rnn.pad_sequence(minibatch_data, padding_value=1)
    minibatch_label = torch.stack(minibatch_label, dim=0)

    return minibatch_data, minibatch_label

def eval_on_test_set(model, test_data, test_labels, bs):
    """ 
    Evaluate the model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
