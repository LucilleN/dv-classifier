"""
This script runs a bidirectional LSTM (Long Short-Term Memory) neural network 
and either trains a new model from scratch, or loads a previously trained model,
then evaluates that model on the testing set. 

Usage:
- Run this script with `python3 lstm.py` to evaluate the trained model on the testing set.
- To retrain the model, run this script with the --train_from_scratch command line argument,
  and optionally specify the following hyperparameters:
    --use_og_data_only: If set, only trains on the original data without any augmented data.
    --use_2_classes: If set, uses 2 classes rather than 3 (collapses class 0 and 1 into one class)
    --n_epochs: Integer representing how many epochs to train the model for
    --batch_size: Integer representing how large each batch should be
    --learning_rate: Float representing the desired learning rate
    --hidden_size: Integer representing the desired dimensions of the hidden layer(s) of the NN
    --num_layers: Integer representing the number of layers 

For further explanations of these flags, consult the README or `utils.py`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from data_loader import load_data
from utils import train_model, evaluate_model, load_data_tensors, parse_command_line_args

SAVE_PATH = 'models/lstm.pt'


class LSTM(nn.Module):
    """
    Implementation of Long Short-Term Memory Neural Network. 
    Set the init parameter `bidirectional` to True to use a bidirectional LSTM (a BLSTM).
    """

    def __init__(self, vocab_size, hidden_size, output_size, num_layers, bidirectional=False):
        """
        Initializes the LSTM with the specified hyperparameters. 
        """
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
        """
        Executes the forward pass of the neural network. This function is called automatically
        when we run `model(...)`.
        """
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
    args = parse_command_line_args()

    # If there's an available GPU, let's use it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_train, labels_train, tok_to_ix = load_data_tensors(
        args.use_og_data_only)

    if args.retrain:
        """
        Initialize model with the specified hyperparameters and architecture
        """
        hidden_size = args.hidden_size
        num_layers = 2 if args.num_layers < 0 else args.num_layers
        vocab_size = len(tok_to_ix)
        output_size = len(np.unique(labels_train))

        model = LSTM(
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            output_size=output_size,
            bidirectional=False)
        model = model.to(device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

        model = train_model(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            data_train=data_train,
            labels_train=labels_train,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            save_path=SAVE_PATH,
            device=device)
    else:
        """
        File was not run with --train_from_scratch, so simply load the model from its saved path
        """
        model = torch.load(SAVE_PATH)

    """
    Whether we're training or just loading the pretrained model, we finish by
    evaluating the model on the testing set.
    """
    evaluate_model(
        model=model,
        tok_to_ix=tok_to_ix,
        use_og_data_only=args.use_og_data_only,
        bs=args.batch_size,
        device=device)
