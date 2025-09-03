# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:06:10 2024

@author: Giovanni Di Liberto
See description in the assignment instructions.
"""

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

# List of notes in order (for single-octave representation)
NOTES = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']

def translate_notes(notes_string, shift):
    translated_notes = []
    for note in notes_string:
        if note in NOTES:
            index = NOTES.index(note)
            new_index = (index + shift) % len(NOTES)
            translated_notes.append(NOTES[new_index])
        else:
            translated_notes.append(note) # Keep the character as is if it's not a note
    return ''.join(translated_notes)

# This inverts each note around a given axis note
def invert_notes(notes_string, axis='F'):
    if axis not in NOTES:
        raise ValueError(f"Axis note '{axis}' is not in the NOTES list.")

    axis_index = NOTES.index(axis)
    inverted_notes = []
    for note in notes_string:
        if note in NOTES:
            idx = NOTES.index(note)
            # Perform inversion around axis_index
            new_idx = (2 * axis_index - idx) % len(NOTES)
            inverted_notes.append(NOTES[new_idx])
        else:
            inverted_notes.append(note)  # Keep the character as is if it's not a note
    return ''.join(inverted_notes)


# Load the input file
with open('inputMelodies.txt', 'r') as file:
    input_melodies = file.readlines()

# Apply 5 different translations and save the results
shifts = [1, 2, 3, 4, 5]
augmented_melodies = []

# This is the axis note around which I will perform pitch inversion (I chose this as it is central in the range of notes)
axis_note = 'F'  


for melody in input_melodies:
    melody_str = melody.strip()

    augmented_melodies.append(melody_str)

    # Pitch Inversion is calculated around the axis note
    inverted_m = invert_notes(melody_str, axis=axis_note)
    augmented_melodies.append(inverted_m)

    # The following code handles the shifting on notes 
    for shift in shifts:
        # Here I Shift the original melody
        shifted_m = translate_notes(melody_str, shift)
        augmented_melodies.append(shifted_m)

        # Here I Shift the inverted melody as well to increase diversity of trainig data 
        inverted_shifted_m = translate_notes(inverted_m, shift)
        augmented_melodies.append(inverted_shifted_m)

# Save all augmented melodies to ano  new file (I updated the file name to differentiate)
with open('inputMelodiesAugmented_updated.txt', 'w') as file:
    for melody in augmented_melodies:
        file.write(melody + '\n')

print("The augmented melodies have been saved to inputMelodiesAugmented_updated.txt")