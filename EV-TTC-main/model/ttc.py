import torch
import lightning as pl
from torch import nn
import hydra
from util import *
from torch.optim.lr_scheduler import OneCycleLR

# Set random seed for reproducibility
pl.seed_everything(1)

class TTCModel(pl.LightningModule):
    """
    PyTorch Lightning module for Time-To-Collision (TTC) prediction.
    This class encapsulates the model, training logic, and evaluation.
    """
    def __init__(self, cfg):
        """
        Initialize the TTC model.
        
        Args:
            cfg: Configuration object with model parameters
        """
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()  # Save configuration for checkpointing

        # Instantiate the model using hydra configuration
        self.model = hydra.utils.instantiate(
            cfg.model_type, cfg=cfg, _recursive_=False, _convert_=None
        )

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input data
            
        Returns:
            TTC output
        """
        return self.model(x)

    def model_step(self, batch, name="train"):
        """
        Performs a single model step (forward pass, loss calculation).
        
        Args:
            batch: Input batch containing data
            name: Step type for logging ('train', 'val', or 'pred')
            
        Returns:
            tuple: (loss, TTC predictions)
        """
        (exp, ttc, mask) = batch
   
        # Get model predictions
        y_hat = self(exp)        
    
        # Calculate TTC loss using Charbonnier loss function with masking
        loss_ttc = charbonnier_loss(ttc - y_hat, alpha=self.cfg.alpha, mask=mask)

        # Log metrics if not in prediction mode
        with torch.no_grad():
            if not name == "pred":
                self.log_dict(
                    {
                        f"{name}_tot": loss_ttc,
                    },
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return loss_ttc, y_hat

    def validation_step(self, batch, batch_idx):
        """
        Validation step - compute loss and visualize results periodically.
        
        Args:
            batch: Input batch
            batch_idx: Index of the current batch
        """
        (exp, ttc, mask) = batch
        loss, ttc_hat = self.model_step(batch, name="val")

        # Periodically visualize validation results
        if batch_idx % self.cfg.viz_interval == 0:
            ttc_plot(
                ttc,
                ttc_hat,
                mask,
                self.logger,
                self.trainer.global_step,
                name="val",
            )

    def training_step(self, batch, batch_idx):
        """
        Training step - compute loss and visualize results periodically.
        
        Args:
            batch: Input batch
            batch_idx: Index of the current batch
            
        Returns:
            loss: The loss value to be optimized
        """
        (exp, ttc, mask) = batch

        loss, ttc_hat = self.model_step(batch, name="train")
        
        # Periodically visualize training results
        with torch.no_grad():
            if batch_idx % self.cfg.viz_interval == 0:
                ttc_plot(
                    ttc,
                    ttc_hat,
                    mask,
                    self.logger,
                    self.trainer.global_step,
                    name="train",
                )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for inference.
        
        Args:
            batch: Input batch
            batch_idx: Index of the current batch
            dataloader_idx: Index of the dataloader
            
        Returns:
            tuple: (model predictions, ground truth, mask)
        """
        (exp, ttc, mask) = batch
        loss, ttc_hat = self.model_step(batch, "pred")

        # Return the first TTC value (likely the most immediate prediction)
        return ttc_hat[:, 0], ttc[:, 0], mask

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            dict: Configuration for optimizer and scheduler
        """
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(self.parameters(), self.cfg.lr)
        
        # Use OneCycleLR scheduler for learning rate adjustment
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.cfg.pct_start,
            cycle_momentum=False,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": schedule, "interval": "step"},
        }
