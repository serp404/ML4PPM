from numpy.lib.histograms import histogramdd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random
from layers import GruLstmLayer, AttentionLayer

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# RNN model based on fanilla LSTM architecture
class LstmModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        n_features, 
        emb_size=128, 
        hid_size=64, 
        num_layers=1, 
        bidirectional=False, 
        embed_features=True
    ):
        super(LstmModel, self).__init__()
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embed_features = embed_features
        self.emb_layer = None

        if self.embed_features:
            self.emb_layer = nn.Linear(
                in_features=n_features,
                out_features=emb_size
            )

        self.vector_size = emb_size if embed_features else n_features

        self.lstm = nn.LSTM(
            input_size=self.vector_size,
            num_layers=self.num_layers,
            hidden_size=self.hid_size,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.head_layers = nn.Sequential(
            nn.Linear(in_features=hid_size, out_features=hid_size),
            nn.Dropout(p=0.2),
            nn.ELU(),
            nn.Linear(in_features=hid_size, out_features=hid_size),
            nn.Dropout(p=0.2),
            nn.ELU(),
            nn.Linear(in_features=hid_size, out_features=vocab_size),
        )


    def forward(self, prefix_batch):
        model_device = next(self.parameters()).device
        batch_size = prefix_batch.shape[0]

        input_batch = None
        if self.embed_features:
            input_batch = self.emb_layer(prefix_batch)
        else:
            input_batch = prefix_batch
        
        layer_dimension = self.num_layers if not self.bidirectional else 2 * self.num_layers
        h0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)
        c0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)

        _, (hn, _) = self.lstm(input_batch, (h0, c0))
        rnn_output = hn[layer_dimension - 1]
        
        logits = self.head_layers(rnn_output)
        return logits


    def get_prefix_embedding(self, prefix_batch):
        model_device = next(self.parameters()).device
        batch_size = prefix_batch.shape[0]

        with torch.no_grad():
            input_batch = None
            if self.embed_features:
                input_batch = self.emb_layer(prefix_batch)
            else:
                input_batch = prefix_batch

            layer_dimension = self.num_layers if not self.bidirectional else 2 * self.num_layers
            h0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)
            c0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)

            _, (hn, _) = self.lstm(input_batch, (h0, c0))

        return hn[layer_dimension - 1]


# RNN model based on LSTM + GRU architecture
class GruLstmModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        n_features, 
        emb_size=128, 
        hid_size=64, 
        num_layers=1, 
        bidirectional=False, 
        embed_features=True
    ):
        super(GruLstmModel, self).__init__()
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embed_features = embed_features
        self.emb_layer = None

        if self.embed_features:
            self.emb_layer = nn.Linear(
                in_features=n_features,
                out_features=emb_size
            )

        self.vector_size = emb_size if embed_features else n_features

        self.rnn_layer = GruLstmLayer(
            input_size=self.vector_size,
            hidden_size=self.hid_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional
        )

        self.head_layers = nn.Sequential(
            nn.Linear(in_features=hid_size, out_features=hid_size),
            nn.Dropout(p=0.2),
            nn.ELU(),
            nn.Linear(in_features=hid_size, out_features=hid_size),
            nn.Dropout(p=0.2),
            nn.ELU(),
            nn.Linear(in_features=hid_size, out_features=vocab_size),
        )


    def forward(self, prefix_batch):
        model_device = next(self.parameters()).device
        batch_size = prefix_batch.shape[0]

        input_batch = None
        if self.embed_features:
            input_batch = self.emb_layer(prefix_batch)
        else:
            input_batch = prefix_batch
        
        layer_dimension = self.num_layers if not self.bidirectional else 2 * self.num_layers
        h0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)
        c0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)

        hn = self.rnn_layer(input_batch, (h0, c0))
        logits = self.head_layers(hn)
        return logits


    def get_prefix_embedding(self, prefix_batch):
        model_device = next(self.parameters()).device
        batch_size = prefix_batch.shape[0]

        with torch.no_grad():
            input_batch = None
            if self.embed_features:
                input_batch = self.emb_layer(prefix_batch)
            else:
                input_batch = prefix_batch

            layer_dimension = self.num_layers if not self.bidirectional else 2 * self.num_layers
            h0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)
            c0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)

            hn = self.rnn_layer(input_batch, (h0, c0))

        return hn


# RNN model based on LSTM with attention architecture
class LstmAttentionModel(nn.Module):
    def __init__(
        self,
        vocab_size, 
        n_features, 
        emb_size=128, 
        hid_size=64, 
        num_layers=1, 
        bidirectional=False, 
        embed_features=True
    ):
        super(LstmAttentionModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embed_features = embed_features
        self.emb_layer = None
        
        if self.embed_features:
            self.emb_layer = nn.Linear(
                in_features=n_features,
                out_features=emb_size
            )

        self.vector_size = emb_size if embed_features else n_features
        
        self.attention = AttentionLayer(
            hidden_size=self.hid_size
        )

        self.lstm = nn.LSTM(
            input_size=self.vector_size,
            num_layers=self.num_layers,
            hidden_size=self.hid_size,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
        self.head_layers = nn.Sequential(
            nn.Linear(in_features=hid_size, out_features=hid_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hid_size * 2, out_features=hid_size * 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hid_size * 2, out_features=vocab_size)
        )


    def forward(self, prefix_batch):
        model_device = next(self.parameters()).device
        batch_size = prefix_batch.shape[0]
        seq_len = prefix_batch.shape[1]

        input_batch = None
        if self.embed_features:
            input_batch = self.emb_layer(prefix_batch)
        else:
            input_batch = prefix_batch

        layer_dimension = self.num_layers if not self.bidirectional else 2 * self.num_layers
        h0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)
        c0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)
        rnn_states, (hn, _) = self.lstm(input_batch, (h0, c0))

        if seq_len > 1:
            prefix = rnn_states[:, :-1, :]
            last = hn[-1]
            return self.head_layers(self.attention(prefix, last))

        return self.head_layers(hn[-1])


    def get_prefix_embedding(self, prefix_batch):
        model_device = next(self.parameters()).device
        batch_size = prefix_batch.shape[0]
        seq_len = prefix_batch.shape[1]

        with torch.no_grad():
            input_batch = None
            if self.embed_features:
                input_batch = self.emb_layer(prefix_batch)
            else:
                input_batch = prefix_batch

            layer_dimension = self.num_layers if not self.bidirectional else 2 * self.num_layers
            h0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)
            c0 = torch.randn((layer_dimension, batch_size, self.hid_size), device=model_device)
            rnn_states, (hn, _) = self.lstm(input_batch, (h0, c0))

            if seq_len > 1:
                prefix = rnn_states[:, :-1, :]
                last = hn[-1]
                return self.attention(prefix, last)

            return hn[-1]
