"""
Training Pipeline for Task 3: Music Generation with AI.

Parses MIDI files using music21, extracts note/chord pitch sequences,
constructs an LSTM Deep Learning architecture in TensorFlow/Keras, and trains the model.
Includes automatic synthetic MIDI data generation for out-of-the-box execution.
"""

import argparse
import json
import logging
import os
import pickle
import random
from typing import List, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_midi_files(output_dir: str, num_files: int = 5) -> None:
    """
    Creates rich synthetic sample MIDI files using music21 if none exist.
    This guarantees the training pipeline can run out-of-the-box anywhere.
    """
    import music21 as m21

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Generating %d sample MIDI files in '%s'...", num_files, output_dir)

    scales = [
        ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
        ["A3", "C4", "D4", "E4", "G4", "A4", "C5", "D5"],
        ["F4", "G4", "A4", "A#4", "C5", "D5", "E5", "F5"],
        ["G3", "B3", "D4", "G4", "B4", "D5", "G5"],
        ["E3", "G3", "B3", "E4", "G4", "B4", "E5"]
    ]

    chords_library = [
        ["C4", "E4", "G4"],
        ["F4", "A4", "C5"],
        ["G4", "B4", "D5"],
        ["A3", "C4", "E4"],
        ["D4", "F4", "A4"]
    ]

    for f_idx in range(num_files):
        stream = m21.stream.Stream()
        stream.append(m21.instrument.Piano())
        stream.append(m21.tempo.MetronomeMark(number=random.choice([100, 110, 120, 130])))

        scale = scales[f_idx % len(scales)]
        
        # Build 64 rhythmic note/chord events
        for _ in range(64):
            event_type = random.random()
            duration = random.choice([0.5, 1.0, 1.5, 2.0])

            if event_type < 0.65:
                # Monophonic Note
                pitch_name = random.choice(scale)
                new_note = m21.note.Note(pitch_name, quarterLength=duration)
                stream.append(new_note)
            elif event_type < 0.90:
                # Polyphonic Chord
                chord_pitches = random.choice(chords_library)
                new_chord = m21.chord.Chord(chord_pitches, quarterLength=duration)
                stream.append(new_chord)
            else:
                # Rest
                new_rest = m21.note.Rest(quarterLength=0.5)
                stream.append(new_rest)

        file_path = os.path.join(output_dir, f"sample_melody_{f_idx + 1}.mid")
        stream.write("midi", fp=file_path)

    logger.info("Sample MIDI files generated successfully in %s.", output_dir)


def extract_notes_from_midis(sample_dir: str) -> List[str]:
    """
    Parses all MIDI files in sample_dir using music21 and extracts pitch tokens.
    """
    import music21 as m21

    midi_files = [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.endswith(".mid") or f.endswith(".midi")
    ]

    if not midi_files:
        generate_sample_midi_files(sample_dir, num_files=5)
        midi_files = [
            os.path.join(sample_dir, f)
            for f in os.listdir(sample_dir)
            if f.endswith(".mid") or f.endswith(".midi")
        ]

    notes = []
    logger.info("Parsing %d MIDI files...", len(midi_files))

    for file_path in midi_files:
        try:
            midi = m21.converter.parse(file_path)
            notes_to_parse = None

            try:
                parts = m21.instrument.partitionByInstrument(midi)
                if parts:
                    notes_to_parse = parts.parts[0].recurse()
                else:
                    notes_to_parse = midi.flat.notes
            except Exception:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, m21.note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, m21.chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))
        except Exception as e:
            logger.warning("Could not parse MIDI file %s: %s", file_path, e)

    logger.info("Extracted total %d note/chord tokens from dataset.", len(notes))
    return notes


def prepare_sequences(
    notes: List[str],
    sequence_length: int = 32
) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    """
    Maps notes to integers and creates sliding window sequences for LSTM training.
    """
    pitchnames = sorted(list(set(notes)))
    n_vocab = len(pitchnames)

    note_to_int = {note: num for num, note in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    if n_patterns == 0:
        raise ValueError(
            f"Not enough note tokens ({len(notes)}) for sequence length {sequence_length}. "
            "Add more MIDI files or decrease sequence_length."
        )

    # Reshape and normalize input for LSTM [samples, time steps, features]
    x = np.reshape(network_input, (n_patterns, sequence_length, 1))
    x = x / float(n_vocab)

    # One-hot encode targets
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(network_output, num_classes=n_vocab)

    return x, y, pitchnames, note_to_int


def build_lstm_model(input_shape: Tuple[int, int], n_vocab: int):
    """
    Constructs a Deep LSTM Neural Network with Dropout and Dense classification.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(256),
        Dropout(0.3),
        BatchNormalization(),
        Dense(128),
        Activation("relu"),
        Dropout(0.2),
        Dense(n_vocab),
        Activation("softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.002),
        metrics=["accuracy"]
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Music Generation Model")
    parser.add_argument("--sample_dir", type=str, default="sample_data", help="Directory containing training MIDI files")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for gradient descent")
    parser.add_argument("--sequence_length", type=int, default=32, help="Sliding lookback window length")
    parser.add_argument("--model_output", type=str, default="music_lstm_model.keras", help="Path to save trained Keras model")
    parser.add_argument("--notes_output", type=str, default="notes.pkl", help="Path to save token vocabulary pickle")
    args = parser.parse_args()

    # Ensure paths are relative to current script directory if not absolute
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, args.sample_dir) if not os.path.isabs(args.sample_dir) else args.sample_dir
    model_output = os.path.join(base_dir, args.model_output) if not os.path.isabs(args.model_output) else args.model_output
    notes_output = os.path.join(base_dir, args.notes_output) if not os.path.isabs(args.notes_output) else args.notes_output

    os.makedirs(sample_dir, exist_ok=True)

    # 1. Extract notes from MIDI files
    notes = extract_notes_from_midis(sample_dir)

    # 2. Save notes list
    with open(notes_output, "wb") as f:
        pickle.dump(notes, f)
    logger.info("Notes token vocabulary saved to %s.", notes_output)

    # 3. Prepare sequences
    X, y, pitchnames, note_to_int = prepare_sequences(notes, sequence_length=args.sequence_length)
    logger.info("Prepared %d training patterns with vocabulary size %d.", X.shape[0], len(pitchnames))

    # Save note mapping JSON for metadata
    mapping_path = os.path.join(base_dir, "mappings.json")
    with open(mapping_path, "w") as f:
        json.dump({"pitchnames": pitchnames, "sequence_length": args.sequence_length}, f, indent=2)

    # 4. Build and train LSTM model
    logger.info("Building LSTM neural network architecture...")
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), n_vocab=len(pitchnames))
    model.summary()

    logger.info("Starting model training for %d epochs...", args.epochs)
    model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    # 5. Save model weights
    model.save(model_output)
    logger.info("Trained model successfully saved to %s!", model_output)


if __name__ == "__main__":
    main()
