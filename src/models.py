"""
CMPSC 448 - Midterm Project
Who Wrote It? Identifying LLMs from Their Responses

the 2 required models, both in PyTorch
    - TextCNN  : convolutions over word windows ( Kim 2014 style )
    - TextBiLSTM : the RNN, a bidirectional LSTM

Ryan Broadhurst, Michael Finn
"""
import torch                    # PyTorch, the deep learning library
import torch.nn as nn           # layers ( Embedding, Conv1d, LSTM, Linear )
import torch.nn.functional as F     # functions without weights ( relu )


class TextCNN(nn.Module):
    """Kim (2014) style CNN for text classification

    tokens -> embedding -> parallel 1-D convolutions with widths {3,4,5} ( n_filters each, ReLU )
    -> global max-pool over time -> concat -> dropout -> linear -> logits
    a width-k filter is an n-gram detector, and max-pooling keeps its strongest match anywhere
    in the response, so the model doesnt care WHERE in the text a pattern shows up
    """

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=128,
                 widths=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)     # word id -> vector ( id 0 is padding, stays zeros )
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k) for k in widths])      # one conv layer per window size
        self.drop = nn.Dropout(dropout)     # randomly zeroes stuff in training so it overfits less
        self.fc = nn.Linear(n_filters * len(widths), n_classes)     # final scores, one per class

    def forward(self, x, lengths=None):
        e = self.emb(x).transpose(1, 2)               # (B, E, T) is the shape Conv1d wants
        mask = (x != 0)     # True where there is a real token ( not padding )
        pooled = []     # strongest filter matches for each window size
        for conv in self.convs:     # go thru each window size
            h = F.relu(conv(e))                        # (B, n_filters, T-k+1)
            k = conv.kernel_size[0]     # width of this filter
            # mask windows that start in padding so they cant win the max-pool
            m = mask[:, : h.size(2)].unsqueeze(1)       # line the mask up with the conv output
            h = h.masked_fill(~m, 0.0)      # padding windows score 0
            pooled.append(h.max(dim=2).values)      # keep the best match anywhere in the text
        return self.fc(self.drop(torch.cat(pooled, dim=1)))     # stick all filter results together, dropout, then class scores


class TextBiLSTM(nn.Module):
    """bidirectional LSTM classifier

    tokens -> embedding -> 1-layer BiLSTM ( hidden H per direction, packed so padding is skipped )
    -> [ masked max-pool over time ; final forward/backward hidden states ]
    -> dropout -> linear -> logits
    """

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=64, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)     # word id -> vector
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)      # reads left->right AND right->left
        self.drop = nn.Dropout(dropout)     # randomly zeroes stuff in training so it overfits less
        self.fc = nn.Linear(4 * hidden, n_classes)      # 4x hidden = 2 directions x ( max-pool + final state )

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))      # embed + dropout
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)        # tells the LSTM to skip the padding
        out, (h_n, _) = self.lstm(packed)       # out = hidden state at every step, h_n = final hidden state of each direction
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))        # unpack back into a normal padded tensor
        mask = (x != 0).unsqueeze(2)        # True where there is a real token
        maxp = out.masked_fill(~mask, -1e4).max(dim=1).values      # (B, 2H) strongest value over time ( padding can never win )
        last = torch.cat([h_n[0], h_n[1]], dim=1)                   # (B, 2H) forward + backward final states side by side
        return self.fc(self.drop(torch.cat([maxp, last], dim=1)))       # glue both read-outs together, dropout, then class scores


def build_model(name, vocab_size, n_classes):
    if name == "cnn":       # the CNN
        return TextCNN(vocab_size, n_classes)
    if name == "lstm":      # the RNN
        return TextBiLSTM(vocab_size, n_classes)
    raise ValueError(name)      # not a model we have
