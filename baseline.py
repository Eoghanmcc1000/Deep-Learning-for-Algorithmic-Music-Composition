import markovify
import math
from typing import Tuple


def load_data(file_path: str) -> str:
    """Read and return the entire contents of a text file.

    Each line in the file is treated as a separate sequence for the Markov model.
    """
    with open(file_path, 'r') as f:
        return f.read()


def train_markov_model(data: str, state_size: int = 4):
    """Train and return a markovify model using newline-delimited sequences."""
    return markovify.NewlineText(data, state_size=state_size)


def generate_sequence(model, min_words: int = 30) -> str:
    """Generate a single sequence from the model.

    Returns None if the model cannot produce a sentence with the requested constraints.
    """
    return model.make_sentence(min_words=min_words)


def calculate_cross_entropy(markov_model, text: str) -> float:
    """Calculate average cross-entropy (in nats) per prediction for given text.

    This iterates over newline-separated sequences, uses the configured state_size
    for the markov chain, and computes -log(p) for each predicted next token.
    """
    total_log_prob = 0.0
    total_predictions = 0
    lines = text.splitlines()

    for line in lines:
        tokens = line.split()
        if len(tokens) <= markov_model.state_size:
            continue

        for i in range(len(tokens) - markov_model.state_size):
            current_state = tuple(tokens[i: i + markov_model.state_size])
            next_word = tokens[i + markov_model.state_size]

            if current_state in markov_model.chain.model:
                transitions = markov_model.chain.model[current_state]
                total_count = sum(transitions.values())
                if next_word in transitions:
                    p = transitions[next_word] / total_count
                else:
                    # OOV transition from this state: treat as tiny probability
                    print("Else 1")
                    p = 1e-12
            else:
                # unseen state entirely
                print("Else 2")
                p = 1e-12

            total_log_prob += -math.log(p)
            total_predictions += 1

    if total_predictions > 0:
        return total_log_prob / total_predictions
    return 0.0


def run_pipeline(input_file: str) -> Tuple[str, float]:
    """Load data, train a Markov model, generate a sequence, and compute cross-entropy.

    Returns the generated sequence and the cross-entropy score.
    """
    data = load_data(input_file)
    model = train_markov_model(data)
    generated_sequence = generate_sequence(model)

    # Print outputs (preserve original behavior)
    print("Generated Sequence:", generated_sequence)
    ce_score = calculate_cross_entropy(model, data)
    print("Cross-Entropy on training data:", ce_score)
    return generated_sequence, ce_score


# default input file (kept for backward compatibility with original script)
input_file = 'finalAssignment_musicDataset/inputMelodiesAugmented_updated.txt'


if __name__ == "__main__":
    run_pipeline(input_file)
