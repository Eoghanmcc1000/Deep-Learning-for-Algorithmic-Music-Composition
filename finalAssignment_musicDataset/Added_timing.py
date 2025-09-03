import os
from mido import MidiFile, MidiTrack, Message, bpm2tempo

# Define the note dictionary
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
}

# Map MIDI note numbers to note names (ignoring octaves)
MIDI_NOTE_TO_NAME = {
    0: 'C', 1: 'c', 2: 'D', 3: 'd', 4: 'E', 5: 'F', 6: 'f',
    7: 'G', 8: 'g', 9: 'A', 10: 'a', 11: 'B'
}

# Function to convert MIDI file to text sequence with timing
def midi_to_text_sequence_with_timing(midi_path):
    midi = MidiFile(midi_path)
    sequence = []
    
    # Default tempo (microseconds per beat)
    current_tempo = 500000  # 120 BPM
    ticks_per_beat = midi.ticks_per_beat
    
    # Accumulate time in ticks
    accumulated_ticks = 0
    
    # To handle multiple tracks, we'll merge them by sorting all messages by their absolute time
    all_msgs = []
    for track in midi.tracks:
        absolute_time = 0
        for msg in track:
            absolute_time += msg.time
            all_msgs.append((absolute_time, msg))
    
    # Sort all messages by their absolute time
    all_msgs.sort(key=lambda x: x[0])
    
    previous_event_tick = 0  # To calculate delta between events
    
    for absolute_tick, msg in all_msgs:
        delta_ticks = absolute_tick - previous_event_tick
        previous_event_tick = absolute_tick
        
        # Convert delta_ticks to milliseconds
        delta_ms = (delta_ticks * current_tempo) / ticks_per_beat / 1000  # Convert to ms
        
        if msg.type == 'set_tempo':
            current_tempo = msg.tempo  # Update current tempo
            continue  # Tempo changes do not correspond to notes or rests
        
        # Handle rests
        if delta_ms > 0:
            # You can choose to represent rests in various ways.
            # Here, we'll encode the duration of the rest.
            rest_token = f"R-{int(delta_ms)}"  # e.g., R-250 for a 250ms rest
            sequence.append(rest_token)
        
        # Handle note_on messages with velocity > 0 (note start)
        if msg.type == 'note_on' and msg.velocity > 0:
            note_number = msg.note % 12
            note = MIDI_NOTE_TO_NAME.get(note_number, '')
            if note:
                # Determine the duration of the note by finding the corresponding note_off
                # This requires searching ahead in the messages
                duration_ticks = 0
                for future_tick, future_msg in all_msgs:
                    if future_tick < absolute_tick:
                        continue
                    if future_msg.type == 'note_off' and future_msg.note == msg.note:
                        duration_ticks = future_tick - absolute_tick
                        break
                    elif future_msg.type == 'note_on' and future_msg.note == msg.note and future_msg.velocity == 0:
                        # Some MIDI files use note_on with velocity 0 as note_off
                        duration_ticks = future_tick - absolute_tick
                        break
                
                if duration_ticks == 0:
                    # If no corresponding note_off is found, assume a default duration
                    duration_ms = 500  # Default to 500ms
                else:
                    duration_ms = (duration_ticks * current_tempo) / ticks_per_beat / 1000  # Convert to ms
                
                note_token = f"{note}-{int(duration_ms)}"  # e.g., C-500 for a C note lasting 500ms
                sequence.append(note_token)
        
        # Optionally, handle other message types if necessary (e.g., tempo changes within the same tick)
    
    # Join the sequence with spaces for readability
    return ' '.join(sequence)

# Function to convert text sequence with timing back to MIDI
def text_sequence_to_midi_with_timing(sequence, output_path):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Default tempo
    current_tempo = 500000  # 120 BPM
    ticks_per_beat = midi.ticks_per_beat
    
    # Split the sequence into tokens
    tokens = sequence.split(' ')
    
    for token in tokens:
        if not token:
            continue
        if token.startswith('R'):
            # Rest token, e.g., R-250
            try:
                duration_ms = int(token.split('-')[1])
            except (IndexError, ValueError):
                duration_ms = 500  # Default duration
            # Convert ms to ticks
            delta_ticks = int((duration_ms * 1000 * ticks_per_beat) / current_tempo)
            track.append(Message('note_off', note=0, velocity=0, time=delta_ticks))
        else:
            # Note token, e.g., C-500
            try:
                note_part, duration_ms = token.split('-')
                duration_ms = int(duration_ms)
            except (IndexError, ValueError):
                note_part = token
                duration_ms = 500  # Default duration
            
            # Find the MIDI note number
            midi_note = None
            for num, name in MIDI_NOTE_TO_NAME.items():
                if name == note_part:
                    midi_note = num + 60  # Assigning to the 5th octave (C5)
                    break
            if midi_note is None:
                print(f"Unknown note: {note_part}, skipping.")
                continue
            
            # Convert ms to ticks
            delta_ticks = int((duration_ms * 1000 * ticks_per_beat) / current_tempo)
            
            # Add note_on and note_off messages
            track.append(Message('note_on', note=midi_note, velocity=64, time=0))
            track.append(Message('note_off', note=midi_note, velocity=64, time=delta_ticks))
    
    midi.save(output_path)

# Directory containing the MIDI files
midi_dir = 'musicDatasetOriginal'

# Directory to store the resulting MIDI files
output_dir = 'musicDatasetSimplified_AddedTiming'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List to store the text sequences
text_sequences = []

# Process each MIDI file in the directory
for file_name in os.listdir(midi_dir):
    if file_name.endswith('.mid'):
        midi_path = os.path.join(midi_dir, file_name)
        text_sequence = midi_to_text_sequence_with_timing(midi_path)
        if text_sequence:  # Check if the sequence is not empty
            text_sequences.append(text_sequence)
        else:
            print(f"No notes found in {file_name}")  # Debugging output

# Write the text sequences to a file
with open("inputMelodiesAugmented_AddedTiming.txt", "w") as file:
    for sequence in text_sequences:
        file.write(sequence + "\n")

# Convert text sequences back to MIDI files
for i, sequence in enumerate(text_sequences):
    output_path = os.path.join(output_dir, f"output_midi_with_timing_{i+1}.mid")
    text_sequence_to_midi_with_timing(sequence, output_path)

print("Text sequences with timing have been written to inputMelodiesAugmented_AddedTiming.txt")
