#!/usr/bin/env python3
"""
Main training script for EXCHANGE Value Network.

Usage:
    # Quick test run
    python scripts/train.py --quick

    # Full training (default: 100 iterations)
    python scripts/train.py --preset medium --iterations 100

    # Resume from checkpoint
    python scripts/train.py --resume runs/experiment/checkpoints/latest.pt

    # Export to ONNX after training
    python scripts/train.py --export-onnx

Example workflow for initial training:
    1. python scripts/train.py --quick  # Test everything works
    2. python scripts/train.py --preset medium --iterations 50 --games 200
    3. python -m src.onnx_export runs/experiment/checkpoints/best.pt models/value_network.onnx
    4. Copy models/value_network.onnx to Godot project
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import Trainer, TrainingConfig
from src.onnx_export import export_to_onnx


def main():
    parser = argparse.ArgumentParser(
        description="Train EXCHANGE value network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Network configuration
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["tiny", "small", "medium", "large"],
        help="Network size preset (default: medium)"
    )

    # Training settings
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Games per iteration (default: 200)"
    )
    parser.add_argument(
        "--bootstrap-games",
        type=int,
        default=1000,
        help="Bootstrap games with random play (default: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (default: 256)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )

    # Workers
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0=auto, default: 0)"
    )

    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/experiment",
        help="Output directory (default: runs/experiment)"
    )

    # MCTS settings
    parser.add_argument(
        "--mcts",
        action="store_true",
        help="Enable MCTS for move selection (Python, slower)"
    )
    parser.add_argument(
        "--mcts-rust",
        action="store_true",
        help="Enable MCTS with Rust simulator (much faster)"
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=100,
        help="MCTS simulations per move (default: 100)"
    )

    # Quick test mode
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run (10 iterations, 20 games each)"
    )

    # Export
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export best model to ONNX after training"
    )

    args = parser.parse_args()

    # Build configuration
    if args.quick:
        config = TrainingConfig(
            network_preset="tiny",
            num_iterations=10,
            games_per_iteration=20,
            bootstrap_games=100,
            bootstrap_epochs=5,
            epochs_per_iteration=3,
            eval_interval=2,
            checkpoint_interval=5,
            output_dir=args.output_dir,
            num_workers=args.workers,
        )
        print("\n*** QUICK TEST MODE ***\n")
    else:
        config = TrainingConfig(
            network_preset=args.preset,
            num_iterations=args.iterations,
            games_per_iteration=args.games,
            bootstrap_games=args.bootstrap_games,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output_dir,
            num_workers=args.workers,
            # MCTS settings
            use_mcts=args.mcts,
            use_mcts_rust=args.mcts_rust,
            mcts_simulations=args.mcts_sims,
        )

    print("=" * 60)
    print("EXCHANGE AI Training")
    print("=" * 60)
    print(f"Network preset: {config.network_preset}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Games per iteration: {config.games_per_iteration}")
    print(f"Bootstrap games: {config.bootstrap_games}")
    if config.use_mcts_rust:
        print(f"MCTS: Rust (fast)")
        print(f"  Simulations/move: {config.mcts_simulations}")
    elif config.use_mcts:
        print(f"MCTS: Python")
        print(f"  Simulations/move: {config.mcts_simulations}")
    else:
        print(f"MCTS: disabled (1-ply evaluation)")
    print(f"Output directory: {config.output_dir}")
    print("=" * 60)

    # Create trainer and run
    trainer = Trainer(config)
    trainer.train(resume_from=args.resume)

    # Export to ONNX if requested
    if args.export_onnx:
        best_checkpoint = Path(config.output_dir) / "checkpoints" / "best.pt"
        onnx_output = Path(config.output_dir) / "models" / "value_network.onnx"
        onnx_output.parent.mkdir(parents=True, exist_ok=True)

        if best_checkpoint.exists():
            print(f"\nExporting to ONNX: {onnx_output}")
            export_to_onnx(str(best_checkpoint), str(onnx_output))
        else:
            print(f"WARNING: Best checkpoint not found at {best_checkpoint}")


if __name__ == "__main__":
    main()
