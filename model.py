# +
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, n_vocab, embedding_size, lstm_size, n_layers = 1, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm_size = lstm_size
        self.lstm_layers = n_layers
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size, n_layers, dropout = dropout)
        self.dense = nn.Linear(lstm_size, n_vocab)
        self.vocab = None
        
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (
            torch.zeros(self.lstm_layers, batch_size, self.lstm_size),
            torch.zeros(self.lstm_layers, batch_size, self.lstm_size)
        )

    def next_word(self, word, prev_state, prev_prob):
        device = self.embedding.weight.device
        
        ind = self.get_token(word)
        x = torch.tensor([[ind]]).to(device)
        
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)
        
        probs = torch.softmax(logits, axis=2)
        probs = probs[0][0]
        probs = probs.cpu().detach().numpy()
        
        return prev_prob + np.log(probs), state
    
    def set_vocab(self, vocab):
        self.vocab = vocab

    def get_token(self, word):
        return self.vocab.stoi[word]
