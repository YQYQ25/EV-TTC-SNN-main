import torch
from omegaconf import OmegaConf
import sys
import os
import argparse
import onnx

# Add the models directory to the path for importing
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))
from ttc import TTCModel

def parse_arguments():
    """Parse command-line arguments for ONNX model export."""
    parser = argparse.ArgumentParser(description='Export TTC model to ONNX format')
    parser.add_argument('--input_model', type=str, required=True, 
                        help='Path to the model checkpoint')
    parser.add_argument('--output_model', type=str, default="inf/ttc_model.onnx",
                        help='Path to save the ONNX model')
    parser.add_argument('--input_dims', type=int, nargs=4, default=[1, 6, 360, 360],
                        help='Input dimensions (batch_size, channels, height, width)')
    return parser.parse_args()

def load_model(checkpoint_path):
    """Load and prepare the model for export."""
    model = (
        TTCModel.load_from_checkpoint(checkpoint_path)
        .cuda()
        .eval()
        .half()
    )
    return model

def export_to_onnx(model, input_shape, output_path):
    """Export the PyTorch model to ONNX format."""
    torch.onnx.export(
        model,                                     # Model being exported
        torch.randn(*input_shape).half().cuda(),   # Example input tensor
        output_path,                               # Output file path 
        input_names=["input"],                     # Model's input names
        output_names=["output"],                   # Model's output names
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        opset_version=17,
        export_params=True,                        # Store the model parameters
        do_constant_folding=True,                  # Optimize constant folding
        verbose=True,                              # Detailed output during export
    )

def clean_onnx_model(model_path):
    """Remove unnecessary inputs from the ONNX graph to optimize the model."""
    # Load the exported ONNX model
    model = onnx.load(model_path)

    # Create a mapping of input names to input objects
    inputs = model.graph.input
    name_to_input = {input.name: input for input in inputs}

    # Remove inputs that are also initializers (constants)
    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    # Save the cleaned model
    onnx.save(model, model_path)
    
def main():
    """Main function to execute the ONNX export process."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load and prepare the model
    model = load_model(args.input_model)
    
    # Use the provided input dimensions
    input_shape = tuple(args.input_dims)
    
    # Export the model to ONNX
    export_to_onnx(model, input_shape, args.output_model)
    
    # Clean up the exported ONNX model
    clean_onnx_model(args.output_model)
    
    print(f"Model successfully exported to {args.output_model}")

if __name__ == "__main__":
    main()
