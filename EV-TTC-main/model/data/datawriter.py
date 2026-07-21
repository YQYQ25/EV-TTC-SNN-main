import time
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter, ModelCheckpoint


class DataWriter(BasePredictionWriter):
    """
    Custom PyTorch Lightning callback for writing model predictions to HDF5 files.
    Stores predicted TTC (Time-to-Collision) values, ground truth, and masks for both
    training and validation sets.
    """
    def __init__(
        self, write_interval="batch", total_data=(100, 100), res=(360, 360), cfg=None
    ):
        """
        Initialize the DataWriter.
        
        Args:
            write_interval: When to write predictions ('batch' or 'epoch')
            total_data: Tuple of (validation_size, training_size)
            res: Resolution of the data as (height, width)
            cfg: Configuration object containing experiment settings
        """
        super().__init__(write_interval)

        # Create HDF5 file with path based on config settings
        output_path = Path(cfg.data_dir) / Path(cfg.ds)
        output_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        self.f_w = h5py.File(
            f"{str(output_path / Path(f'{cfg.task}_{cfg.exp}'))}.h5",
            "w",
        )

        # Create training data group and datasets
        self.train = self.f_w.create_group("train")
        self.train_mask = self.train.create_dataset(
            "mask",
            (total_data[1], res[0], res[1]),
            bool,
            chunks=(1, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),  # Compression settings for efficient storage
        )
        self.train_ttc = self.train.create_dataset(
            "ttc",  # Ground truth TTC values
            (total_data[1], res[0], res[1]),
            np.float16,
            chunks=(1, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        self.train_ttchat = self.train.create_dataset(
            "ttc_hat",  # Predicted TTC values
            (total_data[1], res[0], res[1]),
            np.float16,
            chunks=(1, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),
        )

        # Create validation data group and datasets
        self.val = self.f_w.create_group("val")
        self.val_ttc = self.val.create_dataset(
            "ttc",  # Ground truth TTC values
            (total_data[0], res[0], res[1]),
            np.float16,
            chunks=(1, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        self.val_ttchat = self.val.create_dataset(
            "ttc_hat",  # Predicted TTC values
            (total_data[0], res[0], res[1]),
            np.float16,
            chunks=(1, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        self.val_mask = self.val.create_dataset(
            "mask",
            (total_data[0], res[0], res[1]),
            bool,
            chunks=(1, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),
        )

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """
        Write predictions to HDF5 file at the end of each batch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The Lightning Module
            prediction: Model predictions (TTC hat, TTC ground truth, mask)
            batch_indices: Indices of samples in the batch
            batch: The input batch
            batch_idx: Index of the current batch
            dataloader_idx: Index of the dataloader (0 for validation, 1 for training)
        """
        # For validation dataloader (index 0)
        if dataloader_idx == 0:
            self.val_ttchat[batch_indices[0] : batch_indices[-1] + 1] = (
                prediction[0].half().cpu().detach().numpy()  # Store predictions
            )
            self.val_ttc[batch_indices[0] : batch_indices[-1] + 1] = (
                prediction[1].half().cpu().detach().numpy()  # Store ground truth
            )
            self.val_mask[batch_indices[0] : batch_indices[-1] + 1] = (
                prediction[2][:, 0].cpu().detach().numpy()  # Store masks
            )
        # For training dataloader (index 1)
        else:
            self.train_ttchat[batch_indices[0] : batch_indices[-1] + 1] = (
                prediction[0].half().cpu().detach().numpy()  # Store predictions
            )
            self.train_ttc[batch_indices[0] : batch_indices[-1] + 1] = (
                prediction[1].half().cpu().detach().numpy()  # Store ground truth
            )
            self.train_mask[batch_indices[0] : batch_indices[-1] + 1] = (
                prediction[2][:, 0].cpu().detach().numpy()  # Store masks
            )
