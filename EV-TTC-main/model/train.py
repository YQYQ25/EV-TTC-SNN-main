import hydra  # Hydra for managing configuration
import lightning as pl  # PyTorch Lightning for training management
import torch
from data.datawriter import DataWriter  # For writing output data
from data.ttc_dm import TTCEF_DM  # Data module for Time-To-Collision tasks
from lightning.pytorch.callbacks import ModelCheckpoint  # For saving model checkpoints
from lightning.pytorch.loggers import TensorBoardLogger  # For logging to TensorBoard
from ttc import TTCModel  # The Time-To-Collision model
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg) -> None:
    # Initialize the model using configuration from Hydra
    model = hydra.utils.instantiate(
        cfg.task_model, cfg=cfg, _recursive_=False, _convert_=None
    )

    # Initialize the data module
    dm = hydra.utils.instantiate(
        cfg.dm, cfg=cfg, _recursive_=False, _convert_=None
    )
    
    # Set up TensorBoard logging
    logger = TensorBoardLogger(
        f"{cfg.log_dir}",
        name=f"{cfg.task}_{cfg.ds}",  # Name with task and dataset info
        version=cfg.exp,  # Experiment version
    )

    # Configure model checkpoint saving
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=cfg.save_interval,  # Save every n epochs
        save_on_train_epoch_end=True,
        filename="{epoch:05}",  # Format for saved checkpoint filenames
        monitor="epoch",
        mode="max",
        save_top_k=3,  # Keep the 3 best checkpoints
        auto_insert_metric_name=False,
        save_last=False,
    )
    
    # Configure the PyTorch Lightning trainer
    trainer = pl.Trainer(
        accelerator="cuda",  # Use GPU for training
        logger=logger,
        log_every_n_steps=cfg.log_interval,
        max_epochs=cfg.max_epochs,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        check_val_every_n_epoch=cfg.val_epochs,
        callbacks=[checkpoint_callback],
        precision=cfg.precision,  # Precision for training (16, 32, etc.)
    )
    
    # If a checkpoint path is provided, run prediction
    if cfg.ckpt_path:
        dm.setup(stage="test")

        # Set up a data writer to save predictions
        pred_writer = DataWriter(
            total_data=(len(dm.val_loader), len(dm.train_loader)), res=cfg.res, cfg=cfg
        )
        trainer.callbacks.append(pred_writer)

        # Run prediction using the specified checkpoint
        with torch.no_grad():  # Disable gradient calculation for prediction
            trainer.predict(
                model,
                dataloaders=[dm.val_dataloader(), dm.predict_dataloader()],
                ckpt_path=cfg.ckpt_path,
                return_predictions=False,
            )

    else:
        # Otherwise, train the model
        trainer.fit(
            model=model,
            datamodule=dm,
            ckpt_path=cfg.saved_model,  # Resume from checkpoint if specified
        )
        # Save the final model checkpoint
        os.makedirs(f"{trainer.logger.log_dir}/checkpoints", exist_ok=True)
        trainer.save_checkpoint(f"{trainer.logger.log_dir}/checkpoints/last.ckpt")


if __name__ == "__main__":
    run()
