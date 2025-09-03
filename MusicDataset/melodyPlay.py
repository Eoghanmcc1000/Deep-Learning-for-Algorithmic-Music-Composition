from pydub import AudioSegment
import numpy as np
import simpleaudio as sa

# Define note frequencies (A4 = 440 Hz)
NOTE_FREQUENCIES = {
    'C': 261.63,
    'c': 277.18,  # C#
    'D': 293.66,
    'd': 311.13,  # D#
    'E': 329.63,
    'F': 349.23,
    'f': 369.99,  # F#
    'G': 392.00,
    'g': 415.30,  # G#
    'A': 440.00,
    'a': 466.16,  # A#
    'B': 493.88,
    'R': 0     # Rest
}

# Generate a sine wave for a given frequency and duration in milliseconds
def generate_sine_wave(frequency, duration_ms, sample_rate=44100, amplitude=0.5):
    """
    Generates a sine wave for a specified frequency and duration.

    Parameters:
    - frequency (float): Frequency of the sine wave in Hz.
    - duration_ms (int): Duration of the note in milliseconds.
    - sample_rate (int): Sampling rate in Hz.
    - amplitude (float): Amplitude of the wave.

    Returns:
    - AudioSegment: The generated audio segment.
    """
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    # Normalize to 16-bit PCM range
    wave = (wave * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        wave.tobytes(), 
        frame_rate=sample_rate, 
        sample_width=wave.dtype.itemsize, 
        channels=1
    )
    return audio_segment

# Function to create a sequence of notes with timing
def create_sequence_with_timing(note_sequence):
    """
    Creates an AudioSegment sequence from a list of note-duration tokens.

    Parameters:
    - note_sequence (list): List of strings, each representing a note/rest and its duration (e.g., "C-500").

    Returns:
    - AudioSegment: The complete audio sequence.
    """
    song = AudioSegment.silent(duration=0)  # Start with silence

    for token in note_sequence:
        if not token:
            continue  # Skip empty tokens
        try:
            # Split the token into note and duration
            note, duration_str = token.split('-')
            duration_ms = int(duration_str)
        except ValueError:
            # Handle tokens without a duration (optional)
            print(f"Warning: Token '{token}' is not in the expected 'Note-Duration' format. Using default duration of 500ms.")
            note = token
            duration_ms = 500  # Default duration

        if note == 'R':  # Handle rest
            segment = AudioSegment.silent(duration=duration_ms)
        else:
            frequency = NOTE_FREQUENCIES.get(note)
            if frequency is None:
                print(f"Warning: Note '{note}' is not recognized. Skipping this token.")
                continue  # Skip unrecognized notes
            segment = generate_sine_wave(frequency, duration_ms)
        
        song += segment  # Append the segment to the song

    return song

# Example sequence (Replace this with your actual sequence data)
# The sequence should now contain tokens such as "R-289", etc.
sequence = (
    "R-17448 C-580 R-580 R-31 C-289 R-289 R-16 C-289 R-289 R-16 C-289 R-289 R-16 C-289 R-289 R-16 C-289 R-289 R-16 D-1889 R-1889 R-100 C-144 R-144 R-8 A-289 R-289 R-781 C-144 R-144 R-8 C-144 R-144 R-8 E-289 R-289 R-16 D-144 R-144 R-8 D-144 R-144 R-8 D-144 R-144 R-8 D-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 C-1452 R-1452 R-77 G-580 R-580 R-644 C-289 R-289 R-16 C-144 R-144 R-8 C-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 G-1016 R-1016 R-1738 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 E-144 R-144 R-8 D-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 E-289 R-289 R-16 D-144 R-144 R-8 C-580 R-580 R-31 A-144 R-144 R-8 E-289 R-289 R-1547 E-289 R-289 R-16 A-289 R-289 R-16 G-289 R-289 R-16 E-289 R-289 R-16 C-289 R-289 R-16 D-1016 R-1016 R-54 C-144 R-144 R-8 E-144 R-144 R-8 F-144 R-144 R-8 E-144 R-144 R-8 F-144 R-144 R-8 E-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 C-144 R-144 R-8 C-144 R-144 R-8 C-289 R-289 R-16 E-289 R-289 R-16 D-144 R-144 R-8 D-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 C-1161 R-1161 R-62 A-289 R-289 R-1241 C-289 R-289 R-16 C-144 R-144 R-8 D-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 C-289 R-289 R-16 B-725 R-725 R-39 A-144 R-144 R-8 G-144 R-144 R-8 G-725 R-725 R-39 A-144 R-144 R-1386 C-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 A-289 R-289 R-16 B-289 R-289 R-16 C-144 R-144 R-8 E-725 R-725 R-39 E-144 R-144 R-8 F-144 R-144 R-8 E-289 R-289 R-16 C-434 R-434 R-24 A-289 R-289 R-16 C-289 R-289 R-16 E-144 R-144 R-8 C-144 R-144 R-8 E-434 R-434 R-24 E-144 R-144 R-8 F-144 R-144 R-8 E-289 R-289 R-16 C-434 R-434 R-24 A-289 R-289 R-16 C-289 R-289 R-16 E-144 R-144 R-8 C-144 R-144 R-8 E-434 R-434 R-24 E-144 R-144 R-8 A-144 R-144 R-8 E-289 R-289 R-16 C-434 R-434 R-24 A-192 R-192 R-11 C-192 R-192 R-11 E-192 R-192 R-11 E-192 R-192 R-11 E-289 R-289 R-16 E-289 R-289 R-16 E-580 R-580 R-31 A-289 R-289 R-16 C-144 R-144 R-8 E-289 R-289 R-16 E-434 R-434 R-24 A-289 R-289 R-16 C-144 R-144 R-8 E-289 R-289 R-16 E-434 R-434 R-177 E-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 D-289 R-289 R-16 C-144 R-144 R-8 A-144 R-144 R-8 A-289 R-289 R-16 A-144 R-144 R-8 C-289 R-289 R-16 C-144 R-144 R-8 B-289 R-289 R-16 B-289 R-289 R-16 B-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 C-434 R-434 R-177 C-289 R-289 R-16 G-289 R-289 R-16 E-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 C-289 R-289 R-16 E-289 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 B-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 C-434 R-434 R-24 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 B-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 C-434 R-434 R-24 C-289 R-289 R-16 C-144 R-144 R-8 B-289 R-289 R-16 C-434 R-434 R-24 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 B-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 C-434 R-434 R-24 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 B-289 R-289 R-16 C-289 R-289 R-16 G-289 R-289 R-16 E-144 R-144 R-8 D-144 R-144 R-8 D-289 R-289 R-16 C-434 R-434 R-24 D-289 R-289 R-16 C-289 R-289 R-16 C-289 R-289 R-16 E-289 R-289 R-6445 C-580 R-580 R-31 C-289 R-289 R-16 C-289 R-289 R-16 C-289 R-289 R-16 C-289 R-289 R-16 C-289 R-289 R-16 D-1889 R-1889 R-100 C-144 R-144 R-8 A-289 R-289 R-781 C-144 R-144 R-8 C-144 R-144 R-8 E-289 R-289 R-16 D-144 R-144 R-8 D-144 R-144 R-8 D-144 R-144 R-8 D-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 C-1452 R-1452 R-77 G-580 R-580 R-644 C-289 R-289 R-16 C-144 R-144 R-8 C-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 G-1016 R-1016 R-1738 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 E-144 R-144 R-8 D-289 R-289 R-16 D-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 E-289 R-289 R-16 D-144 R-144 R-8 C-580 R-580 R-31 A-144 R-144 R-8 E-289 R-289 R-1547 E-289 R-289 R-16 A-289 R-289 R-16 G-289 R-289 R-16 E-289 R-289 R-16 C-289 R-289 R-16 D-1016 R-1016 R-54 C-144 R-144 R-8 D-144 R-144 R-8 E-144 R-144 R-8 F-144 R-144 R-8 E-144 R-144 R-8 F-144 R-144 R-8 E-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 C-144 R-144 R-8 C-144 R-144 R-8 C-289 R-289 R-16 E-289 R-289 R-16 D-144 R-144 R-8 D-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 C-1161 R-1161 R-62 A-289 R-289 R-1241 C-289 R-289 R-16 C-144 R-144 R-8 D-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 C-289 R-289 R-16 B-725 R-725 R-39 A-144 R-144 R-8 G-144 R-144 R-8 G-725 R-725 R-39 A-144 R-144 R-1386 C-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 A-289 R-289 R-16 B-289 R-289 R-16 C-144 R-144 R-8 E-725 R-725 R-39 E-144 R-144 R-8 F-144 R-144 R-8 E-289 R-289 R-16 C-434 R-434 R-24 A-289 R-289 R-16 C-289 R-289 R-16 E-144 R-144 R-8 C-144 R-144 R-8 E-434 R-434 R-24 E-144 R-144 R-8 F-144 R-144 R-8 E-289 R-289 R-16 C-434 R-434 R-24 A-289 R-289 R-16 C-289 R-289 R-16 E-144 R-144 R-8 C-144 R-144 R-8 E-434 R-434 R-24 E-144 R-144 R-8 A-144 R-144 R-8 E-289 R-289 R-16 C-434 R-434 R-24 A-192 R-192 R-11 C-192 R-192 R-11 E-192 R-192 R-11 E-192 R-192 R-11 E-192 R-192 R-11 E-192 R-192 R-11 E-289 R-289 R-16 E-289 R-289 R-16 E-580 R-580 R-31 A-289 R-289 R-16 C-144 R-144 R-8 E-289 R-289 R-16 E-434 R-434 R-24 A-289 R-289 R-16 C-144 R-144 R-8 E-289 R-289 R-16 E-434 R-434 R-177 E-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 D-289 R-289 R-16 C-144 R-144 R-8 A-144 R-144 R-8 A-289 R-289 R-16 A-144 R-144 R-8 C-289 R-289 R-16 C-144 R-144 R-8 B-289 R-289 R-16 B-289 R-289 R-16 B-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 C-434 R-434 R-177 C-289 R-289 R-16 G-289 R-289 R-16 E-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 C-289 R-289 R-16 E-289 R-289 R-16 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 B-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 C-434 R-434 R-24 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 B-289 R-289 R-16 C-289 R-289 R-16 D-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 C-434 R-434 R-24 C-289 R-289 R-16 C-144 R-144 R-8 B-289 R-289 R-16 C-434 R-434 R-24 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 B-289 R-289 R-16 C-289 R-289 R-16 B-289 R-289 R-16 C-434 R-434 R-24 C-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 B-289 R-289 R-16 C-289 R-289 R-16 G-289 R-289 R-16 E-144 R-144 R-8 D-144 R-144 R-8 D-289 R-289 R-16 C-434 R-434 R-24 D-289 R-289 R-16 C-289 R-289 R-16 C-289 R-289 R-16 E-289 R-289 R-19455 C-144 R-144 R-8 C-192 R-192 R-11 C-192 R-192 R-11 C-192 R-192 R-11 C-192 R-192 R-11 B-192 R-192 R-11 A-192 R-192 R-11 B-434 R-434 R-24 D-871 R-871 R-47 C-144 R-144 R-8 A-289 R-289 R-1241 C-289 R-289 R-16 D-144 R-144 R-8 E-289 R-289 R-16 D-434 R-434 R-24 C-289 R-289 R-16 A-289 R-289 R-16 D-434 R-434 R-24 E-725 R-725 R-345 G-580 R-580 R-31 E-144 R-144 R-8 D-144 R-144 R-8 C-144 R-144 R-8 C-289 R-289 R-16 A-144 R-144 R-468 C-144 R-144 R-8 C-192 R-192 R-11 C-192 R-192 R-11 C-192 R-192 R-11 C-192 R-192 R-11 B-192 R-192 R-11 A-192 R-192 R-11 D-434 R-434 R-24 E-871 R-871 R-47 C-144 R-144 R-8 A-289 R-289 R-1241 E-144 R-144 R-8 A-289 R-289 R-16 G-289 R-289 R-16 F-144 R-144 R-8 E-289 R-289 R-16 D-144 R-144 R-8 C-144 R-144 R-8 C-144 R-144 R-8 D-434 R-434 R-24 E-289 R-289 R-322 C-289 R-289 R-16 C-289 R-289 R-16 A-144 R-144 R-8 G-144 R-144 R-8 D-144 R-144 R-434 R-24 A-289 R-289 R-16 C-289 R-289 R-16 E-144 R-144 R-8 D-144 R-144 R-8 E-434 R-434 R-24 E-144 R-144 R-8 A-144 R-144 R-8 E-289 R-289 R-16 C-434 R-434 R-24 A-289 R-289 R-16 C-289 R-289 R-16 E-725 R-725 R-39 E-144 R-144 R-8 F-144 R-144 R-8 E-289 R-289 R-16 C-434 R-434 R-24 A-289 R-289 R-16 C-289 R-289 R-16 A-2325 R-2325 R-1"
).split()

# Alternatively, you can load the sequence from a file
# Uncomment the following lines if you have a sequence file
# with open("inputMelodiesAugmented_WithTiming.txt", "r") as file:
#     sequence = [line.strip().split() for line in file.readlines()]
#     sequence = [token for sublist in sequence for token in sublist]  # Flatten the list

print("Parsed Sequence:")
print(sequence)

# Create the sequence with timing
song = create_sequence_with_timing(sequence)

# Save the song to a .wav file
output_wav_path = "output_test_with_timing.wav"
song.export(output_wav_path, format="wav")
print(f"Audio has been exported to {output_wav_path}")

# Play the .wav file using simpleaudio
try:
    wave_obj = sa.WaveObject.from_wave_file(output_wav_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()
except Exception as e:
    print(f"An error occurred while playing the audio: {e}")
