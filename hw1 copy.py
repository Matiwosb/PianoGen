# hw1.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from midi2seq import process_midi_seq, seq2piano  # Import MIDI processing functions
from model_base import ComposerBase  # Import base class
from composer_two import Composer  # Your model implementation

# Load and prepare the dataset
piano_seq = process_midi_seq(datadir='maestro-v1.0.0', n=10000, maxlen=50)
piano_seq_tensor = torch.from_numpy(piano_seq)

batch_size = 32
train_loader = DataLoader(TensorDataset(piano_seq_tensor), shuffle=True, batch_size=batch_size, num_workers=4)

# Initialize and train the model
composer = Composer()
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
composer.to(device)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        batch_seq = batch[0].to(device).long()
        loss = composer.train(batch_seq)
        print(f"Epoch {epoch+1}, Loss: {loss}")

# Save a sample generated sequence as a MIDI file
generated_seq = composer.compose(100)
midi = seq2piano(generated_seq)
midi.write('generated_piano.midi')