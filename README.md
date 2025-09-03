# Deep Learning for Algorithmic Music Composition

A deep learning project focused on generating melodies with GPT-2 style transformer models, supported by different optimisation strategies. 
This project transforms MIDI music files into text sequences, augments the dataset through musical transformations, and trains transformer models to learn and produce musical sequences.

## Project Overview
The project includes:

-Preprocessing and augmentation pipeline
-Several GPT-based model variants
-Nucleus sampling to increase variety in outputs
-Evaluation framework for generated sequences
-A simple Markov Chain baseline for comparison

## Repository Structure

```text
/GPT_2_Music/
├── gpt.py # Core GPT model
├── gpt_adj.py # GPT with nucleus sampling
├── baseline.py # Markov chain baseline
├── evaluate.py # Evaluation metrics
├── finalAssignment_musicDataset/
│   ├── midi2text.py # MIDI → text conversion
│   ├── extractMelodies.py # Melody extraction from MIDI
│   ├── augmentMidiTranslations.py # Basic augmentation methods
│   ├── final_augment.py # Advanced augmentation
│   ├── melodyPlay.py # Playback of melodies
│   ├── added_timing.py # MIDI with timing support
│   ├── inputMelodies.txt # Original dataset
│   ├── inputMelodiesAugmented*.txt # Augmented datasets
│   ├── musicDatasetOriginal/ # Raw MIDI files
│   ├── musicDatasetSimplified/ # Simplified MIDI
│   └── musicDatasetSimplified_AddedTiming/ # Simplified with timing


## Features

### Data Processing & Augmentation

- **MIDI to Text Conversion**  
  Converts melodies into symbolic text where each character represents a note e.g., `C`, `D`, `E`, etc.

- **Augmentation Techniques**  
  - **Pitch Shift**: Transposes melodies by a set interval 
  - **Inversion**: Inversion: Mirrors notes around a pivot (e.g., F) 

### Model Architectures

#### Standard GPT Model (`gpt.py`)

- Transformer with multi-head self-attention 
- Configurable hyperparameters for different datasets  

#### Enhanced GPT with Nucleus Sampling (`gpt_nucleus.py`)

- Top-p (nucleus) sampling for more diverse sequences 
- Reduced repetitive outputs  
- Early stopping to stabalise training 


#### Baseline Markov Chain Model (`baseline.py`)

- State-based probabilistic generation  
- Context window of 4 characters  

### Evaluation Metrics

The project uses multiple evaluation metrics to assess the quality of generated music:

- **Pitch Class Histogram and Entropy**: Measures the distribution diversity of musical notes  
- **Repetition Rate**: Frequency of consecutive notes
- **Pitch Range**: Breadth of pitch values used 
- **Cross-Entropy**: Evaluates the models predictive accuracy  
- **Listening Tests**: Subjective assessment of musical quality  

## Experiments and Results

### Hyperparameter Exploration

Three approaches of increasing complexity were tested. Each step improved results, with the final configuration giving the best performance. (All approaches in code)


- **Approach 3: Final Parameters**

  ```text
  n_embd = 320
  n_head = 4
  n_layer = 5
  batch_size = 128
  max_iters = 9000
  dropout = 0.15
  learning_rate = 1e-4
  ```

### Results Summary

Cross-entropy reduced from 0.6670 to 0.6040
Pitch distribution more consistent with real data
Generated sequences were musically coherent and inventive
Lower repetition than earlier models and baseline

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- markovify (for baseline model)
- mido (for MIDI processing)
- music21 (for music analysis)
- matplotlib (for visualization)


### Usage

1. **Process MIDI Files**

   ```bash
   python midi2text.py
   ```

2. **Augment Dataset**

   ```bash
   python final_augment.py
   ```

3. **Train GPT Model**

   ```bash
   python gpt_adj.py
   ```

4. **Evaluate Outputs**

   ```bash
   python evaluate.py
   ```

5. **Play Generated Melodies**

   ```bash
   python finalAssignment_musicDataset/melodyPlay.py
   ```


**Note:** This project was developed as part of an academic assignment to explore transformer models for music generation for the module CS7CS4-2024/25 (Machine Learning) in Trinity College Dublin.  
