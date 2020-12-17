import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import trange

from data_loader import load_data
from utils import train_model, evaluate_model, load_data_tensors, parse_command_line_args

SAVE_PATH = 'models/gru.pt'


class GRU(nn.Module):
    """
    Implementation of a Gated Recurrent Unit neural network.
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
    args = parse_command_line_args()
    
    # If there's an available GPU, let's use it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_train, labels_train, tok_to_ix = load_data_tensors(args.use_og_data_only)

    if args.retrain:
        """
        Initialize model with the specified hyperparameters and architecture
        """
        hidden_size = args.hidden_size
        num_layers = 1 if args.num_layers < 0 else args.num_layers
        vocab_size = len(tok_to_ix)
        output_size = len(np.unique(labels_train))
        
        model = GRU(
            hidden_size=hidden_size, 
            num_layers=num_layers,
            vocab_size=vocab_size,
            output_size=output_size)
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