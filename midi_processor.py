
import os
import numpy as np
from music21 import converter, instrument, note, chord

def process_midi_files(folder_path):
    sequences = []
    for file in os.listdir(folder_path):
        if file.endswith('.mid'):
            try:
                midi = converter.parse(os.path.join(folder_path, file))
                notes_to_parse = None
                
                try:
                    s2 = instrument.partitionByInstrument(midi)
                    notes_to_parse = s2.parts[0].recurse()
                except:
                    notes_to_parse = midi.flat.notes
                
                notes = []
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
                
                # Create dictionary for notes to integers
                unique_notes = list(set(notes))
                note_to_int = {note: number for number, note in enumerate(unique_notes)}
                
                sequence = [note_to_int[note] for note in notes]
                sequences.append(sequence)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    
    return sequences