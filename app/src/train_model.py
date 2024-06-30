import logging
import warnings

import constants
import model
import utils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Reading file with DNA chains")
sequences = utils.read_dataset_file(constants.DATASET_PATH)

# Take part of all
sequences = sequences[: int(len(sequences) / 20)]

logging.info("Generating dataset")
X_left, X_right, y = utils.generate_dataset(sequences)

logging.info("Making train dataset")
train_dataset = model.DNASequenceDataset(X_left, X_right, y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


logging.info("Starting training process")
NN_model = model.DNASequenceModel()

checkpoint_callback = ModelCheckpoint(
    dirpath="weights", filename="{epoch}-{val_loss:.2f}", monitor="val_loss"
)

trainer = Trainer(
    max_epochs=10,
    devices="auto",
    accelerator="auto",
    callbacks=[checkpoint_callback],
    fast_dev_run=False,
)
trainer.fit(NN_model, train_loader)
