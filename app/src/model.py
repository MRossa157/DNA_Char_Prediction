import numpy as np
import torch
from constants import NUCLEOTIDE_MAPPING
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class DNASequenceModel(LightningModule):
    def __init__(self, num_classes=4, input_dim=15, embedding_dim=10):
        super().__init__()
        # 4 нуклеотиды + 1 для padding_idx=0
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim)
        self.conv1_left = nn.Conv1d(embedding_dim, 16, kernel_size=3, padding=1)
        self.conv1_right = nn.Conv1d(embedding_dim, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 2 * input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_left, x_right):
        x_left = self.embedding(x_left).permute(0, 2, 1)
        x_right = self.embedding(x_right).permute(0, 2, 1)
        x_left = self.conv1_left(x_left)
        x_right = self.conv1_right(x_right)
        x_left = torch.flatten(x_left, start_dim=1)
        x_right = torch.flatten(x_right, start_dim=1)
        x = torch.cat((x_left, x_right), dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        x_left, x_right, y = batch
        y_hat = self(x_left, x_right)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_left, x_right, y = batch
        y_hat = self(x_left, x_right)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class DNASequenceDataset(Dataset):
    def __init__(self, X_left, X_right, y):
        self.mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.X_left = torch.tensor(
            [self.integer_encode(seq) for seq in X_left], dtype=torch.long
        )
        self.X_right = torch.tensor(
            [self.integer_encode(seq) for seq in X_right], dtype=torch.long
        )
        self.y = torch.tensor(
            [self.nucleotide_to_index(nuc) for nuc in y], dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_left[idx], self.X_right[idx], self.y[idx]

    @staticmethod
    def integer_encode(seq):
        return np.array([NUCLEOTIDE_MAPPING[char] for char in seq], dtype=np.int64)

    @staticmethod
    def nucleotide_to_index(nuc):
        return NUCLEOTIDE_MAPPING[nuc]
