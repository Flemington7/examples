import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError as e:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""") from e
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
        Unlike using nn.Embedding that can learn positional encoding during training, this is a fixed positional encoding.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: max_len, 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape: d_model/2
        pe[:, 0::2] = torch.sin(position * div_term)  # boardcast div_term to position, Shape: max_len, d_model/2
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: max_len, 1, d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SpiralPositionalEncoding(nn.Module):
    """
    Positional Encoding tailored for a spiral sequence which is generated from a 2D grid.
    Encodes the 2D grid position (row, column) into embeddings compatible with the transformer.
    """
    def __init__(self, d_model, height, width, dropout=0.1):
        """
        Args:
            d_model: Dimensionality of the embeddings.
            height: Height of the 2D grid.
            width: Width of the 2D grid.
            dropout: Dropout rate.
        """
        super(SpiralPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings for the entire grid
        pe = torch.zeros(height * width, d_model)  # Flattened grid positions, shape: (seq_len, embed_dim)
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

        # Spiral order index mapping
        coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=1)  # Shape: (height*width, 2)
        spiral_idx = self._spiral_indices(height, width)  # Get spiral order indices
        spiral_coords = coords[spiral_idx]  # Rearrange in spiral order, shape: (height*width, 2)

        # Method 1:
        # number_of_positions = height * width
        # position = torch.arange(0, number_of_positions, dtype=torch.float).unsqueeze(1)

        # pe[:, 0::2] = torch.sin(position * div_term)  # Shape: (seq_len, d_model/2)
        # pe[:, 1::2] = torch.cos(position * div_term)

        # Method 2:
        # Encode 2D coordinates into d_model dimensions using parallel operations
        y = spiral_coords[:, 0].float().unsqueeze(1)  # Shape: (seq_len, 1), a.k.a. (height*width, 1)
        x = spiral_coords[:, 1].float().unsqueeze(1)  # Shape: (seq_len, 1)

        # Method 2.1:
        # # Calculate the div_terms for sine and cosine functions
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape: (d_model/2,)

        # # Compute positional encodings for each position in the spiral sequence
        # # This implementation does not coherence with the original paper, 
        # # "We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PE(pos+k) can be represented as a linear function of PE(pos)
        # pe[:, 0::2] = torch.sin(y * div_term)  # Shape: (seq_len, d_model/2)
        # pe[:, 1::2] = torch.cos(x * div_term)  # Shape: (seq_len, d_model/2)

        # Method 2.2:
        # # Calculate the div_terms for sine and cosine functions
        # div_term = torch.exp(torch.arange(0, d_model, 4).float() * (-math.log(10000.0) / d_model))  # Shape: (d_model/4,)

        # # Compute positional encodings for each position in the spiral sequence
        # # This implementation coherence with the original paper
        # pe[:, 0::4] = torch.sin(y * div_term)  # Shape: (seq_len, d_model/4)
        # pe[:, 1::4] = torch.cos(y * div_term)  # Shape: (seq_len, d_model/4)
        # pe[:, 2::4] = torch.sin(x * div_term)  # Shape: (seq_len, d_model/4)
        # pe[:, 3::4] = torch.cos(x * div_term)  # Shape: (seq_len, d_model/4)

        # Method 2.3:
        # Calculate the div_terms for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape: (d_model/2,)

        # Compute positional encodings for each position in the spiral sequence
        # This implementation coherence with the original paper
        pe[:, 0::2] = torch.sin(y * div_term) + torch.sin(x * div_term)  # Shape: (seq_len, d_model/2)
        pe[:, 1::2] = torch.cos(y * div_term) + torch.cos(x * div_term)  # Shape: (seq_len, d_model/2)

        self.register_buffer('pe', pe.unsqueeze(1))  # Shape: (seq_len, 1, d_model)

        # Method 3:
        # learnable positional encodings
        # self.lpe = nn.Embedding(height * width, d_model)

    def forward(self, x):
        r"""Add spiral positional encodings to the input embeddings.
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        # x = x + self.lpe(torch.arange(x.size(0)).to(x.device)).unsqueeze(1)
        return self.dropout(x)
    
    def _spiral_indices(self, height, width):
        """
        Generate spiral order indices for a 2D grid of size height x width.
        Args:
            height: Height of the grid.
            width: Width of the grid.
        Returns:
            torch.Tensor: A tensor of indices representing spiral order, shape: (height*width,)
        Examples:
            >>> spiral_idx = self._spiral_indices(3, 3)
            tensor([4, 3, 6, 7, 8, 5, 2, 1, 0])
        """
        grid = torch.arange(height * width).reshape(height, width)
        spiral_idx = []

        while grid.numel() > 0:
            # Top row
            spiral_idx.extend(grid[0, :].tolist())
            grid = grid[1:, :]  # Remove top row
            if grid.numel() == 0:
                break

            # Right column
            spiral_idx.extend(grid[:, -1].tolist())
            grid = grid[:, :-1]  # Remove right column
            if grid.numel() == 0:
                break

            # Bottom row (reversed)
            spiral_idx.extend(grid[-1, :].flip(0).tolist())
            grid = grid[:-1, :]  # Remove bottom row
            if grid.numel() == 0:
                break

            # Left column (reversed)
            spiral_idx.extend(grid[:, 0].flip(0).tolist())
            grid = grid[:, 1:]  # Remove left column

        # Reverse the spiral order to match the spiral order of the grid
        spiral_idx.reverse()

        return torch.tensor(spiral_idx, dtype=torch.long)

class PositionalEncoding2d(nn.Module):
    # max height / max width
    def __init__(self, d_model, dropout=0.1, height=128, width=128) -> None:
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # pe_1d.shape max_len, 1, tok_emb
        # x.shape  bptt,batch,tok_emb
        # pe_2d.shape tok_emb,max_height,max_width
        w = 4
        h = x.size(0) // w
        odd = x.size(0) - h * w
        # w = h = int(math.sqrt(x.size(0)))
        pe1 = self.pe[:, :h, :w].reshape(x.size(-1), -1) # shape: tok_emb, bptt
        pe2 = self.pe[:, h, :odd].reshape(x.size(-1), -1) # shape: tok_emb, bptt
        pe = torch.cat([pe1, pe2], 1)
        pe = pe.transpose(0, 1).unsqueeze(1) # shape: bptt, 1, tok_emb
        x = x + pe
        return self.dropout(x)

class TransformerModel(nn.Transformer):
    """
    Container module with an encoder, a recurrent or transformer module, and a decoder.
    Args:
        ntoken: Number of tokens in the vocabulary.
        ninp: Number of expected features in the input, a.k.a. embedding dimension.
        nhead: Number of heads in the multiheadattention models.
        nhid: Dimension of the feedforward network model.
        nlayers: Number of recurrent layers.
        dropout: A dropout value of [0, 1).
    """

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, height=128, width=128, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.pos_encoder = SpiralPositionalEncoding(ninp, height=height, width=width, dropout=dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        """Performs the forward pass of the Transformer model.
        Args:
            src (Tensor): The input sequence tensor of shape [sequence length, batch size].
            has_mask (bool): Indicates whether to apply a source mask. Default is True.
        Returns:
            Tensor: Log probabilities of the output tokens, shape [sequence length, batch size, ntoken].
        """
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp) # Shape: (seq_len, batch, embed_dim)
        # src = src.repeat(height)
        src = self.pos_encoder(src) # Add positional encoding to the input embeddings
        output = self.encoder(src, mask=self.src_mask) # Shape: (seq_len, batch, embed_dim)
        output = self.decoder(output) # Shape: (seq_len, batch, ntoken)
        return F.log_softmax(output, dim=-1)
