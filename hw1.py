# composer.py

import torch
import torch.nn as nn
import math
import gdown
import logging
import os
from model_base import ComposerBase
from midi2seq import dim, Event, piano2seq, seq2piano

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
        return x + self.pe[:x.size(0)]

class Composer(nn.Module, ComposerBase):
    def __init__(self, load_trained=False, weights_path=None, d_model=256, nhead=8, num_layers=6, dim_feedforward=512):
        nn.Module.__init__(self)
        ComposerBase.__init__(self, load_trained)
        self.d_model = d_model
        self.embedding = nn.Embedding(dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, dim)
        
        # Determine the device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)
        
        self.to(self.device)
        
        if load_trained:
            self._load_trained_model(weights_path)
        else:
            self.init_weights()

    def _load_trained_weights(self):
        model_path = 'piano_composer_model.pth'
        if not os.path.exists(model_path):
            logging.info('Downloading pre-trained model...')
            url = "https://drive.google.com/drive/folders/1nHWjbBTjFw_-VQjqq3QPhYoDVDagkAvs?usp=sharing"
            gdown.download(url, model_path, quiet=False)
        
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location=self.device))
            logging.info('Loaded pre-trained model.')
        else:
            logging.warning('Failed to load pre-trained model. Initializing with random weights.')
            self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output)
        return output

    def train_model(self, dataloader, num_epochs=10, learning_rate=0.0001):
        nn.Module.train(self)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch[0].to(self.device)  # Move batch to the same device as the model
                loss = self.train_step(batch, criterion, optimizer)
                total_loss += loss
            
            avg_loss = total_loss / len(dataloader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def train_step(self, batch, criterion, optimizer):
        optimizer.zero_grad()
        output = self(batch[:, :-1])
        loss = criterion(output.reshape(-1, dim), batch[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        optimizer.step()
        
        return loss.item()

    def compose(self, n, temperature=1.0):
        nn.Module.eval(self)
        start_token = torch.tensor([[128*2+1]]).to(self.device)
        current_sequence = start_token
        
        with torch.no_grad():
            for _ in range(n-1):
                output = self(current_sequence)
                next_token_logits = output[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                current_sequence = torch.cat([current_sequence, next_token], dim=1)
        
        return current_sequence.squeeze().cpu().numpy()

    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)
        logging.info(f"Model weights saved to {filepath}")