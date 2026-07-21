from torch import nn
import torch


class ConvLayer(nn.Module):
    """
    A custom convolutional layer with optional normalization and activation.
    
    This layer combines convolution, optional batch normalization, and 
    optional activation functions in a single module.
    """
    def __init__(
        self,
        in_channels,      # Number of input channels
        out_channels,     # Number of output channels
        kernel_size,      # Size of the convolutional kernel
        stride=1,         # Stride of the convolution
        activation="ReLU", # Activation function to use (default: ReLU)
        norm=True,        # Whether to use batch normalization
    ):
        super(ConvLayer, self).__init__()

        # If using normalization, bias is not needed
        bias = False if norm else True
        
        # Initialize the convolutional layer
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding='same', # Keep input and output dimensions the same
            bias=bias,
        )

        # Set up activation function if specified
        if activation is not None:
            self.activation = getattr(torch.nn, activation)()
        else:
            self.activation = None

        # Set up batch normalization if requested
        self.norm = norm
        if norm:
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=0.01)

    def forward(self, x):
        """
        Forward pass through the convolutional layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor after convolution, optional normalization and activation
        """
        # Apply convolution
        out = self.conv2d(x)

        # Apply batch normalization if enabled
        if self.norm:
            out = self.norm_layer(out)

        # Apply activation function if specified
        if self.activation is not None:
            out = self.activation(out)

        return out