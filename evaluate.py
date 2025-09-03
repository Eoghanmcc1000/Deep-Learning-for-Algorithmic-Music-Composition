
# uncomment this to process a folder of midi files 

# import os
# import shutil
# from music21 import converter, interval, stream
# from collections import Counter
# import math
# import matplotlib.pyplot as plt

# from finalAssignment_musicDataset.midi2text import text_sequence_to_midi


# def evaluate_midi_file(midi_path):
#     melody = converter.parse(midi_path)
#     timed_notes = melody.flatten().notes

#     # Initialize metrics
#     pitch_range = None
#     repetition_rate = None
#     entropy = None

#     # Calculate Pitch Range
#     pitches = [note.pitch.midi for note in timed_notes if note.isNote]
#     if pitches:
#         pitch_range = max(pitches) - min(pitches)
    
#     # Calculate Note Repetition Rate
#     note_count = len(pitches)
#     consecutive_repetitions = 0
#     for i in range(1, len(pitches)):
#         if pitches[i] == pitches[i - 1]:
#             consecutive_repetitions += 1
#     repetition_rate = (consecutive_repetitions / note_count) if note_count > 0 else 0

#     # Calculate Pitch Class Histogram Entropy
#     pitch_classes = [note.pitch.pitchClass for note in timed_notes if note.isNote]
#     counter = Counter(pitch_classes)
#     total = sum(counter.values())
#     if total > 0:
#         entropy = -sum((count / total) * math.log2(count / total) for count in counter.values())
    
#     return {
#         'pitch_range': pitch_range,
#         'repetition_rate': repetition_rate,
#         'entropy': entropy
#     }, pitch_classes

# def evaluate_all_midis(input_dir):
#     """Evaluates all MIDI files in the specified directory."""
#     metrics = {
#         'pitch_range': [],
#         'repetition_rate': [],
#         'entropy': []
#     }
#     all_pitch_classes = []
    
#     midi_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mid')]
#     if not midi_files:
#         print("No MIDI files found for evaluation.")
#         return metrics, all_pitch_classes
    
#     for midi_file in midi_files:
#         midi_path = os.path.join(input_dir, midi_file)
#         file_metrics, pitch_classes = evaluate_midi_file(midi_path)
        
#         if file_metrics['pitch_range'] is not None:
#             metrics['pitch_range'].append(file_metrics['pitch_range'])
#         if file_metrics['repetition_rate'] is not None:
#             metrics['repetition_rate'].append(file_metrics['repetition_rate'])
#         if file_metrics['entropy'] is not None:
#             metrics['entropy'].append(file_metrics['entropy'])
        
#         all_pitch_classes.extend(pitch_classes)  # Collect pitch classes from each file
#         print(f"Evaluated {midi_file}: {file_metrics}")
    
#     return metrics, all_pitch_classes

# def calculate_average_metrics(metrics):
#     """Calculates the average of each metric."""
#     average_metrics = {}
#     for key, values in metrics.items():
#         if values:
#             average = sum(values) / len(values)
#             average_metrics[key] = average
#         else:
#             average_metrics[key] = None
#     return average_metrics

# def plot_aggregate_histogram(all_pitch_classes):
#     """Plots the aggregate pitch class histogram."""
#     if not all_pitch_classes:
#         print("No pitch classes to plot.")
#         return
    
#     counter = Counter(all_pitch_classes)
#     total = sum(counter.values())
#     pitch_classes = sorted(counter.keys())
#     frequencies = [counter[pc] for pc in pitch_classes]
    
#     plt.figure(figsize=(10, 6))
#     plt.bar(pitch_classes, frequencies, color='skyblue', edgecolor='black')
#     plt.xlabel('Pitch Classes')
#     plt.ylabel('Frequency')
#     plt.title('Aggregate Pitch Class Histogram')
#     # plt.xticks(pitch_classes)  # Ensures all pitch classes are labeled
#     # Map pitch classes to note names
#     note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
#     plt.xticks(pitch_classes, [note_names[pc] for pc in pitch_classes])
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.show()

# def main():
#     input_dir = 'finalAssignment_musicDataset/musicDatasetSimplified'
#     output_dir = 'ground_truth'
    
    
#     # Evaluate all MIDI files
#     metrics, all_pitch_classes = evaluate_all_midis(input_dir)
    
#     # Calculate average metrcs
#     average_metrics = calculate_average_metrics(metrics)
    
#     #Output the results
#     print("\n=== Average Evaluation Metrics ===")
#     if average_metrics['pitch_range'] is not None:
#         print(f"Average Pitch Range: {average_metrics['pitch_range']:.2f} MIDI steps")
#     else:
#         print("No Pitch Range data available.")
    
#     if average_metrics['repetition_rate'] is not None:
#         print(f"Average Note Repetition Rate: {average_metrics['repetition_rate']:.4f}")
#     else:
#         print("No Note Repetition Rate data available.")
    
#     if average_metrics['entropy'] is not None:
#         print(f"Average Pitch Class Histogram Entropy: {average_metrics['entropy']:.4f}")
#     else:
#         print("No Entropy data available.")
    
#     # Plot the aggregate histogram
#     plot_aggregate_histogram(all_pitch_classes)

# if __name__ == "__main__":
#     main()


















# uncomment the following to run evalution on one output
from music21 import converter, interval, stream
from collections import Counter
import math
import matplotlib.pyplot as plt

from finalAssignment_musicDataset.midi2text import text_sequence_to_midi

# sequence = 'R R R R R A A A G G f f f g B B g R a B R a f g g a a C a a G G a a G g G g R d d f g g f g a g g g c c a g c a g f f a g R R g a B'
# sequence = 'R R R R R R R R R R R R R a C d d C C C C d d D R R R R R a a C d d C C C C d d D R R R R R C D d d D a G D D C C C a C a d d G G F G G g a C D d F R R R R R a a C d d C C C C d d D R R R R R a a C d d C C C C d d D R R R R R d d d d D a G D D C C C a C a d d G G F G G g a C D d F a a a G a F d G a F d G a F d C G g G G F a G a F d a G a F d a G a F d d G g G G F a G a F d a G a F d a G a F d a G a F d a G a F d R R G G a D d R R R R R R R R R R R R R R R R R R R R R R R D d D d D D G D C C C a C a G d G g d F G g a C d G F a G a F d G a F d G a F d C G g G G F a G a F d a G a F d a G a F d d G g G G F a G a F d a G a F d a G a F d a G a F d a G a F d f f d f d f f d f f d f f d f g a R R d f f d f d f f d f f d f f d c g f f f g F a a C d d C C C C d d D R R d a g G a a C d d C C C C d d D R R a a a G a F d G a F d G a F d C G g G G F a G a F d a G a F d a G a F d d G g G G F a G a F G a G F d f a G F d G d g G F a G C C d R R R G g a D d '


sequence = 'R R R R R R R R R R R R R d d R d f f d f d R R R R R R R R R R R R R R R R R R R R R R R R R R C B C B C C B C B R C B g g R R R R R R R G G d d R R R R R R R R R R R d d R R R R R C B C C B R R R R R R B C B C C B R R C C B C B C B R R C C E D R R R R R R R R R G A A B C B R C D D D E D R R R C R C D D D R E C C C D R C D R C D D D E C R R R R R D R E C C C C D R R A B C D D R D D C R R R R R E A B A A R B A B A G R D E R D E R D E D E R R D E D A D R E D E R G A A G R R R E A B A G R R G A A'
text_sequence_to_midi(sequence, 'evaluation_files/test.mid')

# Load a melody from a MIDI file
melody = converter.parse('evaluation_files/test.mid')

# melody = converter.parse('finalAssignment_musicDataset/musicDatasetSimplified/output_midi_100.mid')

# Flatten the melody to access notes
timed_notes = melody.flatten().notes

# Extract pitches and calculate pitch range
pitches = [note.pitch.midi for note in timed_notes if note.isNote]
if pitches:
    pitch_range = max(pitches) - min(pitches)
    print(f"Pitch Range: {pitch_range} MIDI steps")
else:
    print("No pitches found.")

# Calculate note repetition rate
note_durations = [note.quarterLength for note in timed_notes if note.isNote]
note_count = len(note_durations)
consecutive_repetitions = 0
for i in range(1, len(pitches)):
    if pitches[i] == pitches[i - 1]:
        consecutive_repetitions += 1
repetition_rate = consecutive_repetitions / note_count
print(f"Note Repetition Rate: {repetition_rate:.4f}")



# Pitch Class Histogram and Entropy
pitch_classes = [p.pitch.pitchClass for p in timed_notes if p.isNote]
counter = Counter(pitch_classes)
total = sum(counter.values())
if total > 0:
    entropy = -sum((count / total) * math.log2(count / total) for count in counter.values())
    print(f"Pitch Class Histogram Entropy: {entropy:.4f}")

    # Plot the histogram
    plt.bar(counter.keys(), counter.values(), tick_label=[str(pc) for pc in counter.keys()])
    plt.xlabel('Pitch Classes')
    plt.ylabel('Frequency')
    plt.title('Pitch Class Histogram of baseline markov chain model')
    plt.show()
else:
    print("No pitch classes to calculate entropy.")
