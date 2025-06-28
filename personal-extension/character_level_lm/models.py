import torch
import torch.nn as nn
import torch.nn.init as init

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, batch_first, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first 
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=batch_first,\
                          dropout=dropout if num_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        # init the embedings
        init.xavier_normal_(self.embeddings.weight)

        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:  # input-to-hidden weights
                init.xavier_normal_(param)
            elif 'weight_hh' in name:  # hidden-to-hidden weights
                init.xavier_normal_(param)
            elif 'bias' in name:  # biases
                init.zeros_(param)
        
        # Initialize output layer
        init.xavier_normal_(self.ffn.weight)
        init.zeros_(self.ffn.bias)
        
    def forward(self, x):
        # x shape (B, seq_len)
        B, seq_len = x.shape

        emb = self.embeddings(x) # (B, seq_len, emb_dim)

        h0 = torch.zeros(self.num_layers, B, self.hidden_dim).to(self.device)
        out, ht = self.rnn(emb, h0) # out shape: (B, seq_len, hidden_dim)

        # out = self.ln(out)
        logits = self.ffn(out) # logits shape: (B, seq_len, vocab_size)
        return logits

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, batch_first, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        # init the embedings
        init.xavier_normal_(self.embeddings.weight)

        # Initialize LSTM weights and biases
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                init.xavier_normal_(param)
            elif 'bias' in name:
                # Initialize all biases to zero
                init.zeros_(param)
                # Set the forget gate bias to 1
                # The gates are ordered: input, forget, cell, output
                n = self.hidden_dim
                param.data[n:2*n].fill_(1.0)
        
        # Initialize output layer
        init.xavier_normal_(self.ffn.weight)
        init.zeros_(self.ffn.bias)
        
    def forward(self, x):
        # x shape (B, seq_len)
        B, seq_len = x.shape

        emb = self.embeddings(x) # (B, seq_len, emb_dim)

        h0 = torch.zeros(self.num_layers, B, self.hidden_dim).to(self.device)
        out, ht = self.rnn(emb, h0) # out shape: (B, seq_len, hidden_dim)

        # out = self.ln(out)
        logits = self.ffn(out) # logits shape: (B, seq_len, vocab_size)
        return logits

class GRULM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, batch_first, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        # init the embedings
        init.xavier_normal_(self.embeddings.weight)

        # Initialize RNN weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:  # input-to-hidden weights
                init.xavier_normal_(param)
            elif 'weight_hh' in name:  # hidden-to-hidden weights
                init.xavier_normal_(param)
            elif 'bias' in name:  # biases
                init.zeros_(param)
        
        # Initialize output layer
        init.xavier_normal_(self.ffn.weight)
        init.zeros_(self.ffn.bias)
        
    def forward(self, x):
        # x shape (B, seq_len)
        B, seq_len = x.shape

        emb = self.embeddings(x) # (B, seq_len, emb_dim)

        h0 = torch.zeros(self.num_layers, B, self.hidden_dim).to(self.device)
        out, ht = self.rnn(emb, h0) # out shape: (B, seq_len, hidden_dim)

        # out = self.ln(out)
        logits = self.ffn(out) # logits shape: (B, seq_len, vocab_size)
        return logits
