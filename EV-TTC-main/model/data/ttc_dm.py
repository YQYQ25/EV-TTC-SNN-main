import torch
from torchvision.transforms import v2
import h5py
import hdf5plugin
import lightning as pl
from torch.utils.data import  DataLoader


class TTCEF_DL(torch.utils.data.Dataset):
    def __init__(self, cfg, train=False, augment=False):
        super().__init__()
        h5_file = h5py.File(cfg.train_file if train else cfg.val_file, "r", libver="latest")
        self.cfg = cfg

        self.exp = h5_file["exp_filts"]
        self.ttc = h5_file["ttc"]
        self.mask = h5_file["mask"]
        self.data_len = len(self.exp)

        self.augment = augment
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=cfg.flip_prob),
            v2.RandomVerticalFlip(p=cfg.flip_prob),
            v2.RandomRotation(degrees=(0, 180))
        ])

    def __getitem__(self, index):
        mask = self.mask[index]
        exp = self.exp[index]
        tot_filts = exp.shape[0]
        ttc = self.ttc[index]

        # Combine all tensors for unified augmentation
        input_ten = torch.cat([
            torch.from_numpy(exp),
            torch.from_numpy(ttc)[None, :, :],  # Add channel dimension
            torch.from_numpy(mask)[None, :, :]
        ]).float()

        # Apply transforms if augmentation is enabled
        if self.augment:
            input_ten = self.transforms(input_ten)

        return (
            input_ten[:tot_filts].float(), # EXP Filts
            input_ten[tot_filts][None, :, :].float(), # TTC
            input_ten[tot_filts+1][None, :, :].bool(), # Mask
        )
       
    def __len__(self):
        return self.data_len




class TTCEF_DM(pl.LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str):
        if stage == "test":
            self.augment = False
        else:
            self.augment = True
        self.train_loader = TTCEF_DL(self.cfg, True, self.augment)
        self.val_loader = TTCEF_DL(self.cfg, False, False)

    def train_dataloader(self):
        return DataLoader(
            self.train_loader,
            batch_size=self.cfg.batch_size,
            shuffle=self.augment,
            num_workers=self.cfg.workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_loader,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.train_loader,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.workers,
            persistent_workers=True,
            pin_memory=True,
        )
