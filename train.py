# +
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchtext
from torchtext import data

import argparse

import numpy as np
import pickle

import os

from model import LSTMModel
 
import spacy
from spacy.symbols import ORTH


# -

def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


def predict(device, net, init_words, vocab, top_k = 5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    
    words = init_words.copy()
    
    for w in words:
        ix = torch.tensor([[vocab.stoi[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(vocab.itos[choice])
    
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(vocab.itos[choice])

    print(' '.join(words))


def train(flags):
    my_tok = spacy.load('en')

    def spacy_tok(x):
        return [tok.text for tok in my_tok.tokenizer(x)]

    TEXT = data.Field(lower=True, tokenize=spacy_tok)
    
    dataset = torchtext.datasets.LanguageModelingDataset(flags.train_file, TEXT)
    dataset[0].text = dataset[0].text[::-1]
    
    if flags.custom_embeddings:
        custom_embeddings = torchtext.vocab.Vectors(name=os.path.abspath(flags.custom_embeddings))
        TEXT.build_vocab(dataset, vectors=custom_embeddings)
    else:
        TEXT.build_vocab(dataset, vectors="glove.6B.300d")
    
    weight_matrix = TEXT.vocab.vectors
    vocab = TEXT.vocab
    
    os.makedirs(flags.save_dir, exist_ok=True)
    with open(os.path.join(flags.save_dir, 'vocab.pkl'), 'wb') as vocab_file:
        pickle.dump(vocab, vocab_file)
    
    train_iter = data.BPTTIterator(
        dataset,
        batch_size=flags.batch_size,
        bptt_len=flags.seq_size, # this is where we specify the sequence length
        device=torch.device("cuda:0"),
        repeat=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_vocab, emb_size = weight_matrix.shape

    net = LSTMModel(n_vocab, emb_size, flags.lstm_size, flags.lstm_layers)
    net.embedding.weight.data.copy_(weight_matrix)
    net.set_vocab(vocab)

    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, flags.learning_rate)

    iteration = 0
    
    for e in range(flags.n_epoch):
        state_h, state_c = net.zero_state(flags.batch_size)

        # Transfer data to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        for batch in train_iter:
            x, y = batch.text, batch.target

            iteration += 1

            # Tell it we are in training mode
            net.train()

            # Reset all gradients
            optimizer.zero_grad()

            # Transfer data to GPU
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()

            # Perform back-propagation
            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(net.parameters(), flags.gradients_norm)

            # Update the network's parameters
            optimizer.step()

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, flags.n_epoch),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

            if iteration % 1000 == 0:
                predict(device, net, ['the end'], vocab, top_k=2)
                torch.save(net.state_dict(), os.path.join(flags.save_dir, 'model-last.pth'))


if __name__ == '__main__':
    np.random.seed(2018)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/whitman/input.txt',
                       help='file containing train texts')
    parser.add_argument('--save_dir', type=str, default='model',
                       help='directory to store checkpointed models')
    parser.add_argument('--custom_embeddings', type=str, default='',
                       help='path to custom embeddings')
    parser.add_argument('--lstm_size', type=int, default=1000,
                       help='size of LSTM hidden state')
    parser.add_argument('--lstm_layers', type=int, default=3,
                       help='number of layers in the LSTM')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='minibatch size')
    parser.add_argument('--seq_size', type=int, default=1,
                       help='sequence length')
    parser.add_argument('--n_epoch', type=int, default=20,
                       help='number of epochs')
    parser.add_argument('--gradients_norm', type=float, default= 5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='learning rate')
    
    args = parser.parse_args()
    train(args)
