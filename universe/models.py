from torch import nn

from universe.constants import PADDING_VALUE


class BiLSTM(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        num_layers, 
        input_size, 
        num_classes, 
        vocab_size, 
        dropout=0.2
    ):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PADDING_VALUE)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout
            )
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        lstm_h_n, _ = self.lstm(self.embedding(x))
        out = self.fc(lstm_h_n[:, -1, :])
        return out