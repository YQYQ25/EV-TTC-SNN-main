import torch
import lightning as pl
from layers.conv_lay import *
from layers.aspp import ASPP
from torch import nn
pl.seed_everything(1)  # Set random seed for reproducibility


class EVSlim(pl.LightningModule):
    """
    A lightweight neural network model for event-based vision tasks using PyTorch Lightning.
    The architecture consists of an encoder, an ASPP (Atrous Spatial Pyramid Pooling) module,
    and a decoder network.
    """
    def __init__(self, cfg):
        """
        Initialize the EVSlim model.
        
        Args:
            cfg: Configuration object containing model hyperparameters
        """
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing
        self.cfg = cfg
        self.in_channels = getattr(cfg, "in_channels", getattr(cfg, "input", 6))

        # Encoder network: series of convolutional layers with increasing channels
        self.enc = nn.Sequential(*[
            ConvLayer(
            in_channels=self.in_channels if i == 0 else cfg.enc_channels[i-1],  # First layer uses configurable input channels
            out_channels=cfg.enc_channels[i],  # Output channels for this layer
            kernel_size=(cfg.enc_k_size[i], cfg.enc_k_size[i]),  # Square kernel of configured size
            activation=cfg.act,  # Activation function from config
            norm=cfg.norm  # Normalization method from config
            ) for i in range(len(cfg.enc_channels))
        ])
        
        # ASPP module for multi-scale feature extraction with different dilation rates
        self.aspp = nn.Sequential(*[
            ASPP(
            in_channels=cfg.enc_channels[-1] if i == 0 else cfg.aspp_channels[i-1],  # First ASPP takes encoder output
            out_channels=cfg.aspp_channels[i],  # Output channels for this ASPP module
            atrous_rates=tuple(cfg.rate),  # Dilation rates for atrous convolutions
            ) for i in range(len(cfg.aspp_channels))
        ])
        
        # Decoder network: series of convolutional layers with decreasing channels
        self.dec = nn.Sequential(*[
            ConvLayer(
            in_channels=cfg.aspp_channels[-1] if i == 0 else cfg.dec_channels[i-1],  # First layer uses ASPP output
            out_channels=cfg.dec_channels[i],  # Output channels for this layer
            kernel_size=(cfg.dec_k_size[i], cfg.dec_k_size[i]),  # Square kernel of configured size
            activation=cfg.act if i < len(cfg.dec_channels) - 1 else None,  # No activation for final layer
            norm=cfg.norm if i < len(cfg.dec_channels) - 1 else None  # No normalization for final layer
            ) for i in range(len(cfg.dec_channels))
        ])

    def forward(self, x):
        """
        Forward pass of the EVSlim model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through encoder, ASPP and decoder
        """
        x = self.enc(x)  # Pass input through encoder
        x = self.aspp(x)  # Pass encoder output through ASPP module
        x = self.dec(x)   # Pass ASPP output through decoder

        return x
