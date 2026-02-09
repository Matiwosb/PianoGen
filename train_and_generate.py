# train_and_generate.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import glob
import os
import logging
from composer_one import Composer
from midi2seq import piano2seq, seq2piano

logging.basicConfig(level=logging.INFO)

def process_midi_files(file_paths, seq_length):
    sequences = []
    for file_path in file_paths:
        try:
            seq = piano2seq(file_path)
            for i in range(0, len(seq) - seq_length, seq_length // 2):
                sequences.append(seq[i:i+seq_length])
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
    
    if not sequences:
        raise ValueError("No valid sequences were extracted from the MIDI files.")
    
    # Convert list of sequences to a single numpy array
    sequences_array = np.array(sequences)
    
    # Convert numpy array to PyTorch tensor
    return torch.from_numpy(sequences_array).long()

# Load and process MIDI files
midi_dir = '/Users/matiwosbirbo/PianoGen/maestro-v1.0.0 2/'
midi_files = glob.glob(os.path.join(midi_dir, '*.midi'))

if not midi_files:
    midi_files = glob.glob(os.path.join(midi_dir, '*.mid'))  # Try .mid extension

if not midi_files:
    raise FileNotFoundError(f"No MIDI files found in directory: {midi_dir}")

logging.info(f"Found {len(midi_files)} MIDI files.")

seq_length = 512
try:
    training_data = process_midi_files(midi_files, seq_length)
    logging.info(f"Processed {len(training_data)} sequences from MIDI files.")
except ValueError as e:
    logging.error(str(e))
    raise

# Create DataLoader
batch_size = 32
dataset = TensorDataset(training_data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the Composer
composer = Composer(load_trained=False)

# Check if MPS (Apple Silicon GPU) is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     logging.info("Using Apple Silicon GPU (MPS)")
# else:
#     device = torch.device("cpu")
#     logging.info("Using CPU")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

# Move the model to the appropriate device
composer = composer.to(device)

# Train the model
num_epochs = 5
learning_rate = 0.001

composer.train_model(data_loader, num_epochs, learning_rate)

# Generate music
generated_sequence_length = 1000
generated_sequence = composer.compose(generated_sequence_length)

# Convert the generated sequence to MIDI and save
midi_obj = seq2piano(generated_sequence)
midi_obj.write('generated_piano_piece.mid')

logging.info("Music generation complete. Output saved as 'generated_piano_piece.mid'")