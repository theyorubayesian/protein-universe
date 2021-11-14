"""
Written by: Akintunde 'theyorubayesian' Oladipo
14/Nov/2021
"""
from torch import nn

from universe.constants import PADDING_VALUE


class BiLSTM(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        num_layers, 
        embedding_size, 
        num_classes, 
        vocab_size, 
        dropout=0.3
    ):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PADDING_VALUE)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout
            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    """
    # TODO: Check why this stalls training
    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed_input = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)

        output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=PADDING_VALUE)

        out_forward = output[range(len(output)), lengths-1, :self.hidden_size]
        out_reverse = output[:, 0, self.hidden_size:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        
        out = self.fc(out_reduced)
        out = torch.squeeze(out, 1)
        out = torch.sigmoid(out)
        return out
    """

    # NOTE: This may be performing more computations than it should
    # Use `pack_padded_sequence` and `pad_packed_sequence`
    def forward(self, x, h = None):
        emb = self.embedding(x) # [batch size, sent_len, emb dim]
        lstm_out, *_ = self.lstm(emb)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out
