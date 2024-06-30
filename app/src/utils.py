import logging

from tqdm import tqdm


def read_dataset_file(filepath):
    with open(filepath, "r") as file:
        data = file.read().split(">")[1:]
        sequences = [
            seq.split("\n", 1)[1].replace("\n", "").replace("N", "") for seq in data
        ]
        sequences = [seq for seq in sequences if set(seq) <= {"A", "C", "G", "T"}]
        return sequences


def generate_dataset(sequences, window_size=15):
    X_left, X_right, y = [], [], []
    threshold = 2 * window_size + 1
    for seq in tqdm(sequences):
        n = len(seq)
        if n >= threshold:
            for i in range(window_size, n - window_size):
                X_left.append(seq[i - window_size : i])
                X_right.append(seq[i + 1 : i + 1 + window_size])
                y.append(seq[i])
        else:
            logging.debug(f"Skip sequence {seq} cause too small (<{threshold})")

    return X_left, X_right, y
