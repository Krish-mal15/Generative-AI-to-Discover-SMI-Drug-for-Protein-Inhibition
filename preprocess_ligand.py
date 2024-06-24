# 1. Ligand Vocabulary
# 2. Ligand One-Hot Encoding
# 3. Embedding
# 4. Positional Encoding

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math

bind_csv_dataset = pd.read_csv('main_bind_dataset.csv')
smiles_sequences = bind_csv_dataset['smiles'].to_list()

smiles_dictionary = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'I': 8, '(': 9, ')': 10, '[': 11,
                     ']': 12, '=': 13, '#': 14, ':': 15, '.': 16, '1': 17, '2': 18, '3': 19, '4': 20, '5': 21, '6': 22,
                     '7': 23, '8': 24, '9': 25, '@': 26, '@@': 27, '/': 28, '\\': 29, 'c': 30, 'n': 31, 'o': 32,
                     's': 33, '+': 34, '-': 35, 'H': 36, '': 37, 'l': 38}

vocab_size = 38
n_dims = 128


def positional_encoding_function(self, x):
    seq_length = x.size(1)
    hidden_size = x.size(2)
    positional_encoding = torch.zeros(seq_length, hidden_size)
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    positional_encoding = positional_encoding.unsqueeze(0).to(x.device)
    return x + positional_encoding


def embed_ligand():
    embedding = nn.Embedding(vocab_size, n_dims)



