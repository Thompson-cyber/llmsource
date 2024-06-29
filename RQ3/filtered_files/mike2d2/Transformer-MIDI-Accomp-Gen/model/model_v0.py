# -------------------------------------------------------------------------------
# Define Transformer Model Architecture
# -------------------------------------------------------------------------------

"""
The Transformer class defines the architecture of the transformer model:
- input_dim  specifies the dimensionality of the input token sequence,
- output_dim specifies the dimensionality of the output token sequence,
- d_model    specifies the dimensionality of the transformer layers,
- nhead      specifies the number of attention heads in the transformer layers,
- num_layers specifies the number of transformer layers,
- dropout    specifies the dropout probability.
The PositionalEncoding class defines the positional encoding used by the transformer layers:
- d_model    specifies the dimensionality of the transformer layers,
- dropout    specifies the dropout probability,
- max_len    specifies the maximum length of the input sequence.
init_weights method initializes the weights of the embedding and output layers.
forward      method processes the input token sequence and generates the output token sequence.
The input token sequence is first embedded using an embedding layer,
    then processed using a positional encoding layer,
    and then passed through the transformer layers.
The output of the transformer layers is then passed through a linear layer to generate the output token sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # useful to apply activation function like F.relu() (ReLU, softmax, sigmoid) to
#  output or calculate a loss function like F.cross_entropy() during training.
import math
import numpy as np


class DrumTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers=2, dropout=0.1):
        self.is_training = True
        super(DrumTransformer, self).__init__()
        self.src_embedding = nn.Embedding(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(output_dim, d_model)
        self.src_pos_encoder = PositionalEncoding(d_model, dropout)
        self.tgt_pos_encoder = PositionalEncoding(d_model, dropout)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.decoder = nn.Linear(d_model, output_dim)

    def init_weights(self):
        init_range = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=init_range)

    def forward(self, src, target):
        src = self.src_embedding(src)
        src = self.src_pos_encoder(src)
        target = self.tgt_embedding(target)
        target = self.tgt_pos_encoder(target)

        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(target, memory)
        logits = self.decoder(output)
        probs = F.softmax(logits)

        return probs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += self.pe[:x.size(0), :]
        return self.dropout(x)


# -------------------------------------------------------------------------------
# Loads the data from a file,
#   converts each token sequence to a PyTorch LongTensor,
#   and stores them as a list.
# -------------------------------------------------------------------------------
import torch.utils.data as data


class PercussionDataset(data.Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                tokens = [int(token) for token in line.strip().split()]
                self.data.append(tokens)

    # Returns a single sequence at a time.
    def __getitem__(self, index):
        return torch.LongTensor(self.data[index])

    # Returns the total number of sequences in the dataset.
    def __len__(self):
        return len(self.data)


# batch_size = 64
# train_dataset = PercussionDataset('./.txt')

# PyTorch DataLoader that loads training data. Created using DataLoader class and should be set up to load MIDI data.
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -------------------------------------------------------------------------------
# Initialize the transformer model
# -------------------------------------------------------------------------------

import torch.optim as optim  # Useful for optimizer and lr scheduler (Adam, SGD, Adagrad)

num_drum_types = 6
vocab_size = num_drum_types + 2  # +2 for silence and shift tokens

input_dim = 318  # size of your vocabulary, i.e., the number of unique tokens in your MIDI data.
output_dim = vocab_size  # size of vocabulary as well since model will be generating token sequences of same length as input.
d_model = 512  # this determines the dimensionality of the model and is usually set to be in the range of 128-512,
#  depending on the complexity of the task. You can experiment with different values
nhead = 8  # number of attention heads in the model and can be set to 4-8 for most tasks.
num_layers = 6  # number of layers in the encoder stack, which can be set to be around 4-8.
dropout = 0.1  # probability of an element being zeroed in output of a layer, which can be set to 0.1-0.3 to prevent overfitting.
learning_rate = 1e-4  # lr that paper specifies
lr_step_size = 5  # number of epochs after which the learning rate is reduced, which can be set to around 5-10.
lr_gamma = 0.5  # factor by which lr is reduced, which can be set to 0.5-0.9 depending on the rate of convergence.
num_epochs = 300  # number that paper specifies
device = 'cuda'  # CPU or GPU that model should be trained on. If GPU available, set this to "cuda" to enable GPU acceleration.

# Initialize the transformer model
# model = DrumTransformer(input_dim, output_dim, d_model, nhead, num_layers, dropout)

# Define optimizer and learning rate scheduler
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

# # Define loss function
# criterion = nn.CrossEntropyLoss()


# # -------------------------------------------------------------------------------
# # Train model
# # -------------------------------------------------------------------------------
# def mask_data(src, column, instrument_percentage=0.4, timestep_percentage=.2):
#     length = len(src)
#     mask_length = int(length * instrument_percentage)
#     instrument_indices = torch.randperm(length)[:mask_length]
#     mask_length = int(length * timestep_percentage)
#     timestep_indicies = torch.randperm(length)[:mask_length]
#     src_cpy = src.copy()
#     src_cpy[instrument_indices, column] = 0
#     src_cpy[timestep_indicies] = 0

#     return src_cpy


# for epoch in range(num_epochs):
#     running_loss = 0.0
#     model.train()
#     masked_data = mask_data(train_dataset.data, np.random.randint(0, input_dim))
#     shuffled_idxs = torch.randperm(len(masked_data))
#     for i, batch in enumerate(masked_data[shuffled_idxs]):
#         # Get the inputs and targets
#         inputs, targets = batch[:, :input_dim], batch[:, input_dim]
#         inputs, targets = inputs.to(device), targets.to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(inputs, targets[:-1])

#         # Compute the loss
#         loss = criterion(outputs.reshape(-1, output_dim), targets[1:].reshape(-1))

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()

#         # Update the running loss
#         running_loss += loss.item()

#         # Print statistics every n mini-batches
#         n = 100
#         if i % n == n - 1:
#             print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / n:.3f}')
#             running_loss = 0.0

#     # Update the learning rate
#     scheduler.step()

# # Save the trained model
# torch.save(model, 'path/to/your/trained/model.pt')

# another possible model generated by chatgpt
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src):
        src = self.norm(src)
        for layer in self.layers:
            src = layer(src)
        return src

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, tgt, memory, tgt_mask, memory_mask):
        tgt = self.norm(tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return tgt

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, src_vocab_size, tgt_vocab_size):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        memory = self.encoder(self.src_embedding(src))
        output = self.decoder(self.tgt_embedding(tgt), memory, tgt_mask, src_mask)
        output = self.fc(output)
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
