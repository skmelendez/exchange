"""
ONNX Export Utility for EXCHANGE Value Network

Exports trained PyTorch models to ONNX format for inference in C#/Godot.
ONNX Runtime provides cross-platform, high-performance inference.

Usage:
    python -m src.onnx_export checkpoints/best.pt models/value_network.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort

from .game_state import INPUT_CHANNELS, BOARD_SIZE
from .value_network import ExchangeValueNetwork


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 17,
    verify: bool = True,
) -> None:
    """
    Export PyTorch model to ONNX format.

    Args:
        model_path: Path to PyTorch checkpoint (.pt file)
        output_path: Output path for ONNX model (.onnx file)
        opset_version: ONNX opset version (17 for good compatibility)
        verify: Whether to verify the exported model
    """
    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if "network_state_dict" in checkpoint:
        # Full training checkpoint
        network_config = checkpoint.get("network_config", {})
        if network_config:
            network = ExchangeValueNetwork.from_config(network_config)
        else:
            network = ExchangeValueNetwork()  # Default config
        network.load_state_dict(checkpoint["network_state_dict"])
    elif isinstance(checkpoint, ExchangeValueNetwork):
        # Direct model save
        network = checkpoint
    else:
        # Assume it's a state dict
        network = ExchangeValueNetwork()
        network.load_state_dict(checkpoint)

    network.eval()
    print(f"Model parameters: {network.count_parameters():,}")

    # Create dummy input
    dummy_input = torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    # Dynamic axes for batch size
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    # Export to ONNX
    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        network,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )

    if verify:
        verify_onnx_model(output_path, dummy_input, network)

    print("Export complete!")


def verify_onnx_model(
    onnx_path: str,
    sample_input: torch.Tensor,
    pytorch_model: torch.nn.Module,
    tolerance: float = 1e-5,
) -> None:
    """
    Verify ONNX model produces same outputs as PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        sample_input: Sample input tensor
        pytorch_model: Original PyTorch model
        tolerance: Maximum allowed difference
    """
    print("\nVerifying ONNX model...")

    # Check ONNX model validity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model structure is valid")

    # Create ONNX Runtime session
    providers = ['CPUExecutionProvider']
    if 'CoreMLExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CoreMLExecutionProvider')  # Use CoreML on Mac
    session = ort.InferenceSession(onnx_path, providers=providers)

    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(sample_input).numpy()

    # Get ONNX output
    onnx_input = {session.get_inputs()[0].name: sample_input.numpy()}
    onnx_output = session.run(None, onnx_input)[0]

    # Compare outputs
    max_diff = np.abs(pytorch_output - onnx_output).max()
    print(f"  Max output difference: {max_diff:.2e}")

    if max_diff < tolerance:
        print("  Verification PASSED!")
    else:
        print(f"  WARNING: Output difference exceeds tolerance ({tolerance})")

    # Performance test
    print("\nPerformance test (100 inferences)...")
    import time

    # Batch inference
    batch_input = np.random.randn(100, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32)

    start = time.time()
    for i in range(100):
        session.run(None, {session.get_inputs()[0].name: batch_input[i:i+1]})
    elapsed = time.time() - start

    print(f"  Single inference: {elapsed*10:.2f}ms average")

    # Batch inference test
    start = time.time()
    session.run(None, {session.get_inputs()[0].name: batch_input})
    batch_elapsed = time.time() - start

    print(f"  Batch (100) inference: {batch_elapsed*1000:.2f}ms total ({batch_elapsed*10:.3f}ms/position)")


def get_model_info(onnx_path: str) -> dict:
    """Get information about an ONNX model."""
    model = onnx.load(onnx_path)

    info = {
        "opset_version": model.opset_import[0].version,
        "producer": model.producer_name,
        "inputs": [],
        "outputs": [],
    }

    for inp in model.graph.input:
        shape = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in inp.type.tensor_type.shape.dim]
        info["inputs"].append({
            "name": inp.name,
            "shape": shape,
        })

    for out in model.graph.output:
        shape = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in out.type.tensor_type.shape.dim]
        info["outputs"].append({
            "name": out.name,
            "shape": shape,
        })

    return info


def main():
    parser = argparse.ArgumentParser(description="Export EXCHANGE value network to ONNX")
    parser.add_argument("input", type=str, help="Path to PyTorch checkpoint")
    parser.add_argument("output", type=str, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(
        args.input,
        args.output,
        opset_version=args.opset,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
