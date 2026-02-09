import torch
import torch.nn as nn
import math
import logging
import os
from typing import List, Optional
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model_base import ComposerBase
from midi2seq import dim, Event, piano2seq, seq2piano

@dataclass
class ComposerConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    max_len: int = 5000
    dropout: float = 0.1

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Composer(nn.Module, ComposerBase):
    def __init__(self, config: ComposerConfig, load_trained: bool = False, weights_path: Optional[str] = None):
        nn.Module.__init__(self)
        ComposerBase.__init__(self, load_trained)
        self.config = config
        
        self.embedding = nn.Embedding(dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_len, config.dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            config.d_model, config.nhead, config.dim_feedforward, 
            config.dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.num_layers)
        self.fc_out = nn.Linear(config.d_model, dim)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)
        
        if load_trained:
            self._load_trained_model(weights_path)
        else:
            self.init_weights()

    def _load_trained_model(self, weights_path: Optional[str]):
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
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embedding(src) * math.sqrt(self.config.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output)
        return output

    def train_model(self, dataloader: DataLoader, num_epochs: int = 10, learning_rate: float = 0.0001):
        nn.Module.train(self)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch[0].to(self.device)
                loss = self.train_step(batch, criterion, optimizer)
                total_loss += loss
            
            avg_loss = total_loss / len(dataloader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            scheduler.step()

    def train_step(self, batch: torch.Tensor, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        optimizer.zero_grad()
        output = self(batch[:, :-1])
        loss = criterion(output.reshape(-1, dim), batch[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def compose(self, n: int, temperature: float = 1.0) -> List[int]:
        nn.Module.eval(self)
        start_token = torch.tensor([[128*2+1]]).to(self.device)
        current_sequence = start_token
        
        for _ in range(n-1):
            output = self(current_sequence)
            next_token_logits = output[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            current_sequence = torch.cat([current_sequence, next_token], dim=1)
        
        return current_sequence.squeeze().cpu().tolist()

    def save_weights(self, filepath: str):
        torch.save(self.state_dict(), filepath)
        logging.info(f"Model weights saved to {filepath}")