import logging

import numpy as np
from constants import NUCLEOTIDE_MAPPING
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


def one_hot_encode(seq):
    new_seq = []
    for nucleotide in tqdm(seq, desc="One hot encoding"):
        encoded_chain = ""
        if len(nucleotide) == 1:
            encoded_chain += str(NUCLEOTIDE_MAPPING[nucleotide])
        else:
            for char_nucleotide in nucleotide:
                encoded_chain += str(NUCLEOTIDE_MAPPING[char_nucleotide])
        new_seq.append(encoded_chain)

    return new_seq


if __name__ == "__main__":
    X_left = [
        "ATGGAGATCCCTGTG",
        "TGGAGATCCCTGTGC",
        "GGAGATCCCTGTGCC",
        "GAGATCCCTGTGCCT",
        "AGATCCCTGTGCCTG",
        "GATCCCTGTGCCTGT",
        "ATCCCTGTGCCTGTG",
        "TCCCTGTGCCTGTGC",
    ]

    y = ["C", "C", "T", "G", "T", "G", "C", "A", "G", "C", "C", "G", "T", "C", "T"]

    new_X = one_hot_encode(X_left)

    new_Y = one_hot_encode(y)

    print(new_X, new_Y)
