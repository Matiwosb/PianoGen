import torch
import torch.nn as nn
import math
import logging
import os
from model_base import ComposerBase  # Ensure this file exists and is imported correctly
from midi2seq import dim, Event, piano2seq, seq2piano  # Ensure these are defined or imported correctly

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input x
        return x + self.pe[:x.size(0)]

class Composer(nn.Module, ComposerBase):
    def __init__(self, load_trained=False, weights_path=None, d_model=256, nhead=8, num_layers=6, dim_feedforward=512):
        nn.Module.__init__(self)
        ComposerBase.__init__(self, load_trained)
        self.d_model = d_model
        self.embedding = nn.Embedding(dim, d_model)  # Embedding layer
        self.pos_encoder = PositionalEncoding(d_model)  # Positional encoding
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)  # Transformer encoder
        self.fc_out = nn.Linear(d_model, dim)  # Final output layer
        
        # Device setup: MPS for Apple Silicon, fallback to CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logging.info("Using MPS (GPU) for computation")
        else:
            self.device = torch.device("cpu")
            logging.info("MPS not available. Using CPU")

        # Move the model to the device (GPU or CPU)
        self.to(self.device)
        
        if load_trained:
            self._load_trained_model(weights_path)
        else:
            self.init_weights()

    def _load_trained_model(self, weights_path):
        # Load a pre-trained model, if available
        if weights_path and os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, map_location=self.device))
            logging.info(f"Loaded weights from {weights_path}")
        else:
            model_path = 'piano_composer_model.pth'
            if os.path.exists(model_path):
                logging.info('Loading model from default file...')
                self.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                logging.warning('No pre-trained model found. Initializing with random weights.')
                self.init_weights()

    def init_weights(self):
        # Initialize weights
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Forward pass: input -> embedding -> pos encoding -> transformer encoder -> output layer
        src = self.embedding(src) * math.sqrt(self.d_model)  # Scale by sqrt(d_model)
        src = self.pos_encoder(src)  # Add positional encoding
        output = self.transformer_encoder(src)  # Transformer encoder
        output = self.fc_out(output)  # Final linear layer
        return output

    def train_model(self, dataloader, num_epochs=10, learning_rate=0.0001):
        # Training loop
        self.train()  # Set model to training mode
        criterion = nn.CrossEntropyLoss()  # Loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)  # Adam optimizer
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch[0].to(self.device)  # Move input data to the same device as the model
                loss = self.train_step(batch, criterion, optimizer)
                total_loss += loss
            
            avg_loss = total_loss / len(dataloader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def train_step(self, batch, criterion, optimizer):
        optimizer.zero_grad()  # Zero out gradients
        output = self(batch[:, :-1])  # Forward pass
        loss = criterion(output.reshape(-1, dim), batch[:, 1:].reshape(-1))  # Compute loss
        loss.backward()  # Backpropagation
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)  # Gradient clipping
        optimizer.step()  # Update model parameters
        
        return loss.item()

    def compose(self, n, temperature=1.0):
        # Composition logic: generate sequence step by step
        self.eval()  # Set model to evaluation mode
        start_token = torch.tensor([[128*2+1]]).to(self.device)  # Starting token on the same device
        current_sequence = start_token
        
        with torch.no_grad():  # Disable gradient computation
            for _ in range(n-1):
                output = self(current_sequence)  # Forward pass
                next_token_logits = output[:, -1, :] / temperature  # Adjust by temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)  # Sample next token
                current_sequence = torch.cat([current_sequence, next_token], dim=1)  # Append next token to sequence
        
        return current_sequence.squeeze().cpu().numpy()  # Return composed sequence

    def save_weights(self, filepath):
        # Save the model's weights
        torch.save(self.state_dict(), filepath)
        logging.info(f"Model weights saved to {filepath}")