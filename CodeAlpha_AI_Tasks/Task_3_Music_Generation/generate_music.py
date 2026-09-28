"""
Music Generation Inference Engine for Task 3: Music Generation with AI.

Loads trained LSTM weights and notes vocabulary to generate novel polyphonic/monophonic
musical compositions and writes them to playable standard MIDI files (.mid).
"""

import argparse
import datetime
import logging
import os
import pickle
import random
from typing import List, Optional

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sample_with_temperature(probabilities: np.ndarray, temperature: float = 1.0) -> int:
    """
    Applies temperature scaling to output probability distribution to control randomness/creativity.
    """
    temperature = max(0.1, min(temperature, 2.0))
    probs = np.asarray(probabilities).astype("float64")
    
    # Avoid log(0)
    probs = np.clip(probs, 1e-8, 1.0)
    log_probs = np.log(probs) / temperature
    exp_probs = np.exp(log_probs)
    scaled_probs = exp_probs / np.sum(exp_probs)

    # Sample from multinomial distribution
    sampled_idx = int(np.random.choice(len(scaled_probs), p=scaled_probs))
    return sampled_idx


def generate_notes_with_model(
    model_path: str,
    notes_path: str,
    num_notes: int = 100,
    sequence_length: int = 32,
    temperature: float = 0.8
) -> List[str]:
    """
    Generates a sequence of note/chord string tokens using the trained LSTM model.
    """
    import tensorflow as tf

    logger.info("Loading notes vocabulary from %s...", notes_path)
    with open(notes_path, "rb") as f:
        notes = pickle.load(f)

    pitchnames = sorted(list(set(notes)))
    n_vocab = len(pitchnames)
    note_to_int = {note: num for num, note in enumerate(pitchnames)}
    int_to_note = {num: note for num, note in enumerate(pitchnames)}

    # Build input sequences to select a seed
    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        seq_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])

    if not network_input:
        # Fallback synthetic seed
        seed_pattern = [random.randint(0, n_vocab - 1) for _ in range(sequence_length)]
    else:
        start_idx = np.random.randint(0, len(network_input) - 1)
        seed_pattern = network_input[start_idx]

    logger.info("Loading model from %s...", model_path)
    model = tf.keras.models.load_model(model_path)

    prediction_output = []
    pattern = list(seed_pattern)

    logger.info("Generating %d novel notes with temperature %.2f...", num_notes, temperature)
    for _ in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)[0]
        index = sample_with_temperature(prediction, temperature=temperature)
        result = int_to_note[index]

        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output


def generate_notes_algorithmic_fallback(num_notes: int = 100) -> List[str]:
    """
    Algorithmic Markov/Harmony fallback generator that creates melodic sequences
    without needing heavy TensorFlow model training.
    """
    logger.info("Running algorithmic harmony music generator...")
    scale = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5"]
    chords = ["0.4.7", "5.9.0", "7.11.2", "9.0.4", "2.5.9"]

    output = []
    curr_pitch_idx = random.randint(0, len(scale) - 1)

    for _ in range(num_notes):
        roll = random.random()
        if roll < 0.70:
            # Stepwise or small leap
            step = random.choice([-2, -1, 0, 1, 2])
            curr_pitch_idx = max(0, min(len(scale) - 1, curr_pitch_idx + step))
            output.append(scale[curr_pitch_idx])
        elif roll < 0.90:
            # Harmonic Chord
            output.append(random.choice(chords))
        else:
            # Octave shift
            output.append(random.choice(["C4", "G4", "E4", "A4"]))

    return output


def create_midi_from_notes(prediction_output: List[str], output_file: str) -> str:
    """
    Converts note/chord string tokens into music21 Stream and writes a standard MIDI file.
    """
    import music21 as m21

    offset = 0.0
    output_notes = []

    for pattern in prediction_output:
        # Pattern is a chord (contains dot or digits)
        if ("." in pattern) or pattern.isdigit():
            try:
                notes_in_chord = pattern.split(".")
                chord_notes = []
                for current_note in notes_in_chord:
                    if current_note.isdigit():
                        new_note = m21.note.Note(int(current_note))
                    else:
                        new_note = m21.note.Note(current_note)
                    new_note.storedInstrument = m21.instrument.Piano()
                    chord_notes.append(new_note)
                new_chord = m21.chord.Chord(chord_notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            except Exception:
                new_note = m21.note.Note("C4")
                new_note.offset = offset
                output_notes.append(new_note)

        # Pattern is a single note
        else:
            try:
                new_note = m21.note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = m21.instrument.Piano()
                output_notes.append(new_note)
            except Exception:
                new_note = m21.note.Note("C4")
                new_note.offset = offset
                output_notes.append(new_note)

        # Increment offset for rhythmic spacing
        offset += random.choice([0.5, 0.75, 1.0])

    midi_stream = m21.stream.Stream(output_notes)
    midi_stream.insert(0, m21.tempo.MetronomeMark(number=120))
    midi_stream.insert(0, m21.instrument.Piano())

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    midi_stream.write("midi", fp=output_file)
    logger.info("MIDI file generated and saved successfully: %s", output_file)
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate Novel AI Music from Trained LSTM Model")
    parser.add_argument("--model", type=str, default="music_lstm_model.keras", help="Path to trained model")
    parser.add_argument("--notes", type=str, default="notes.pkl", help="Path to notes vocabulary file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated MIDI")
    parser.add_argument("--num_notes", type=int, default=120, help="Number of notes/chords to generate")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature (0.2=conservative, 1.2=creative)")
    parser.add_argument("--fallback", action="store_true", help="Force algorithmic fallback generator")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, args.model) if not os.path.isabs(args.model) else args.model
    notes_path = os.path.join(base_dir, args.notes) if not os.path.isabs(args.notes) else args.notes
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"ai_music_composition_{timestamp}.mid")

    use_model = (
        not args.fallback
        and os.path.exists(model_path)
        and os.path.exists(notes_path)
    )

    if use_model:
        try:
            logger.info("Using trained Neural LSTM model for generation...")
            prediction_output = generate_notes_with_model(
                model_path=model_path,
                notes_path=notes_path,
                num_notes=args.num_notes,
                temperature=args.temperature
            )
        except Exception as e:
            logger.warning("Neural model inference encountered an issue (%s). Falling back to algorithmic generator.", e)
            prediction_output = generate_notes_algorithmic_fallback(args.num_notes)
    else:
        if not args.fallback:
            logger.info("No trained model weights found at %s. Using algorithmic harmony generator.", model_path)
        prediction_output = generate_notes_algorithmic_fallback(args.num_notes)

    saved_path = create_midi_from_notes(prediction_output, output_file)
    print(f"\n[+] SUCCESS: Generated AI Music Composition!")
    print(f"[>] Output File: {saved_path}")
    print(f"[*] Total Note/Chord Events: {len(prediction_output)}")


if __name__ == "__main__":
    main()
