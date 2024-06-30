import numpy as np
import torch
import utils
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class DNASequenceModel(LightningModule):
    def __init__(self, num_classes=4, input_dim=15):
        super().__init__()
        self.conv1_left = nn.Conv1d(4, 16, kernel_size=3, padding=1)
        self.conv1_right = nn.Conv1d(4, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 2 * input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_left, x_right):
        x_left = self.conv1_left(x_left.permute(0, 2, 1))
        x_right = self.conv1_right(x_right.permute(0, 2, 1))
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
        """
        Args:
            X_left (List[str]): Список строк, представляющих левые окна последовательностей ДНК.
            X_right (List[str]): Список строк, представляющих правые окна последовательностей ДНК.
            y (List[str]): Список символов, представляющих центральные нуклеотиды, которые модель должна предсказать.
        """
        self.X_left = X_left
        self.X_right = X_right
        self.y = y

        # One hot encoding
        self.X_left = utils.one_hot_encode(self.X_left)
        self.X_right = utils.one_hot_encode(self.X_right)
        self.y = utils.one_hot_encode(self.y)

        # list -> numpy
        # self.X_left = np.asarray(self.X_left, dtype="<U1")
        # self.X_right = np.asarray(self.X_right, dtype="<U1")
        # self.y = np.asarray(self.y, dtype="<U1")

        # to tensors
        print("Начинаем перевод в тензоры")
        self.X_left = torch.tensor(self.X_left, dtype=torch.float32)
        print("Закончили левые")
        self.X_right = torch.tensor(self.X_right, dtype=torch.float32)
        print("Закончили правые")
        self.y = torch.tensor(self.y, dtype=torch.long)
        print("Закончили лейблы")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_left[idx], self.X_right[idx], self.y[idx]
