import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize


def charbonnier_loss(error, alpha=0.45, mask=None):
    """
    Calculates Charbonnier loss, a differentiable variant of L1 loss.
    
    Args:
        error: The error/difference to calculate loss on
        alpha: Exponent parameter for the Charbonnier function (default: 0.45)
        mask: Optional mask to only consider certain pixels
    
    Returns:
        The calculated Charbonnier loss (scalar)
    """
    if mask is not None:
        valid = mask.bool() & torch.isfinite(error)
        safe_error = torch.where(valid, error, torch.zeros_like(error))
        charbonnier = (safe_error**2.0 + 1e-5**2.0) ** alpha
        valid_float = valid.float()
        denom = torch.sum(valid_float, dim=(1, 2, 3))
        per_sample = torch.sum(charbonnier * valid_float, dim=(1, 2, 3)) / denom.clamp_min(1.0)
        keep = denom > 0
        loss = per_sample[keep].mean() if torch.any(keep) else safe_error.sum() * 0.0
    else:
        error = torch.where(torch.isfinite(error), error, torch.zeros_like(error))
        charbonnier = (error**2.0 + 1e-5**2.0) ** alpha
        loss = torch.mean(charbonnier)
    return loss

def ttc_plot(y, y_hat, mask, logger, step, name="train"):
    """
    Visualizes Time-To-Collision (TTC) ground truth and predictions and logs to tensorboard.
    
    Args:
        y: Ground truth TTC values
        y_hat: Predicted TTC values
        mask: Mask indicating valid regions
        logger: Logger object for visualization
        step: Current training step
        name: Prefix for the log name (default: "train")
    """
    torch._dynamo.disable(recursive=False)  # Disable torch dynamo for visualization
    with torch.no_grad():
        idx_batch = -1  # Use the last batch element
        y = y.float()
        y_hat = y_hat.float()

        # Create colormap for visualization
        cmap = cm.viridis
        norm = Normalize(
            vmin=torch.quantile(y[idx_batch, 0, :, :], 0.2),
            vmax=torch.quantile(y[idx_batch, 0, :, :], 0.75),
        )

        # Apply colormap to ground truth and predictions
        s1 = cmap(norm((y[idx_batch, 0, :, :]).detach().cpu().numpy()))[:, :, :3]
        s2 = cmap(norm((y_hat[idx_batch, 0, :, :]).detach().cpu().numpy()))[:, :, :3]

        # Create side-by-side comparison image
        res_img = np.zeros((3, y_hat.shape[2], y_hat.shape[3] * 2))
        res_img[:, :, : y_hat.shape[3]] = (
            s1.transpose(2, 0, 1) * mask[idx_batch].detach().cpu().numpy()
        )
        res_img[:, :, y_hat.shape[3] :] = (
            s2.transpose(2, 0, 1) * mask[idx_batch].detach().cpu().numpy()
        )

        # Log the visualization to tensorboard
        logger.experiment.add_image(f"{name}_ttc_img", res_img, global_step=step)
