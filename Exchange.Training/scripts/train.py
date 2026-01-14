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
        help="Checkpoint path to load (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/experiment",
        help="Output directory (default: runs/experiment)"
    )

    # Simulator modes
    parser.add_argument(
        "--rust",
        action="store_true",
        help="Enable Rust 1-ply evaluation (fastest, for 'easy' mode training)"
    )
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

    # Hybrid mode
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid training: mix fast 1-ply with quality MCTS games"
    )
    parser.add_argument(
        "--hybrid-ratio",
        type=float,
        default=0.2,
        help="Fraction of games to use MCTS in hybrid mode (default: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--hybrid-sims",
        type=int,
        default=50,
        help="MCTS simulations per move in hybrid mode (default: 50)"
    )

    # Asymmetric mode (teacher-student)
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="Enable asymmetric training: MCTS (teacher) vs 1-ply (student) in same game"
    )
    parser.add_argument(
        "--asymmetric-sims",
        type=int,
        default=50,
        help="MCTS simulations for teacher side in asymmetric mode (default: 50)"
    )

    # MCTS exploration settings
    parser.add_argument(
        "--mcts-noise",
        type=float,
        default=0.5,
        help="Dirichlet noise weight for MCTS (0-1, higher = more random, default: 0.5)"
    )
    parser.add_argument(
        "--mcts-alpha",
        type=float,
        default=0.15,
        help="Dirichlet alpha for MCTS (lower = spikier noise, default: 0.15)"
    )
    parser.add_argument(
        "--no-noise-decay",
        action="store_true",
        help="Disable noise decay (keep noise constant throughout training)"
    )
    parser.add_argument(
        "--noise-min",
        type=float,
        default=0.15,
        help="Minimum noise epsilon after decay (default: 0.15)"
    )
    parser.add_argument(
        "--noise-decay-iters",
        type=int,
        default=100,
        help="Iterations to decay noise from --mcts-noise to --noise-min (default: 100)"
    )

    # Outcome values
    parser.add_argument(
        "--draw-penalty",
        type=float,
        default=0.8,
        help="Draw penalty (default: 0.8)"
    )
    parser.add_argument(
        "--loss-penalty",
        type=float,
        default=1.0,
        help="Loss penalty (default: 1.0, symmetric with win)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=0.10,
        help="Boredom penalty per position repetition (default: 0.10)"
    )

    # Combo bonus for attacks during Royal Decree
    parser.add_argument(
        "--no-combo-bonus",
        action="store_true",
        help="Disable combo bonus for attacks during Royal Decree"
    )
    parser.add_argument(
        "--combo-bonus-per-attack",
        type=float,
        default=0.05,
        help="Bonus per combo attack during Royal Decree (default: 0.05)"
    )
    parser.add_argument(
        "--combo-bonus-max",
        type=float,
        default=0.3,
        help="Maximum combo bonus cap (default: 0.3)"
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
            # Hybrid mode (also available in quick test)
            use_hybrid=args.hybrid,
            hybrid_mcts_ratio=args.hybrid_ratio,
            hybrid_mcts_simulations=args.hybrid_sims,
            # Asymmetric mode (also available in quick test)
            use_asymmetric=args.asymmetric,
            asymmetric_mcts_simulations=args.asymmetric_sims,
            # MCTS exploration (wire to quick mode too)
            mcts_dirichlet_epsilon=args.mcts_noise,
            mcts_dirichlet_alpha=args.mcts_alpha,
            mcts_noise_decay=not args.no_noise_decay,
            mcts_noise_min_epsilon=args.noise_min,
            mcts_noise_decay_iterations=args.noise_decay_iters,
            # Outcome values
            draw_penalty=args.draw_penalty,
            loss_penalty=args.loss_penalty,
            # Combo bonus
            combo_bonus_enabled=not args.no_combo_bonus,
            combo_bonus_per_attack=args.combo_bonus_per_attack,
            combo_bonus_max=args.combo_bonus_max,
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
            # Simulator mode
            use_rust=args.rust,
            use_mcts=args.mcts,
            use_mcts_rust=args.mcts_rust,
            mcts_simulations=args.mcts_sims,
            # Hybrid mode
            use_hybrid=args.hybrid,
            hybrid_mcts_ratio=args.hybrid_ratio,
            hybrid_mcts_simulations=args.hybrid_sims,
            # Asymmetric mode
            use_asymmetric=args.asymmetric,
            asymmetric_mcts_simulations=args.asymmetric_sims,
            # MCTS exploration
            mcts_dirichlet_epsilon=args.mcts_noise,
            mcts_dirichlet_alpha=args.mcts_alpha,
            mcts_noise_decay=not args.no_noise_decay,
            mcts_noise_min_epsilon=args.noise_min,
            mcts_noise_decay_iterations=args.noise_decay_iters,
            # Outcome values
            draw_penalty=args.draw_penalty,
            loss_penalty=args.loss_penalty,
            repetition_penalty=args.repetition_penalty,
            # Combo bonus
            combo_bonus_enabled=not args.no_combo_bonus,
            combo_bonus_per_attack=args.combo_bonus_per_attack,
            combo_bonus_max=args.combo_bonus_max,
        )

    print("=" * 60)
    print("EXCHANGE AI Training")
    print("=" * 60)
    print(f"Network preset: {config.network_preset}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Games per iteration: {config.games_per_iteration}")
    print(f"Bootstrap games: {config.bootstrap_games}")
    # Helper to format noise info
    def noise_info():
        base = f"epsilon={config.mcts_dirichlet_epsilon:.2f}, alpha={config.mcts_dirichlet_alpha:.2f}"
        if config.mcts_noise_decay:
            return f"{base} (decays to {config.mcts_noise_min_epsilon:.2f} over {config.mcts_noise_decay_iterations} iters)"
        return f"{base} (no decay)"

    if config.use_asymmetric:
        print(f"Mode: ASYMMETRIC (MCTS teacher vs 1-ply student)")
        print(f"  MCTS simulations/move: {config.asymmetric_mcts_simulations}")
        print(f"  MCTS noise: {noise_info()}")
    elif config.use_hybrid:
        oneply_pct = int((1 - config.hybrid_mcts_ratio) * 100)
        mcts_pct = int(config.hybrid_mcts_ratio * 100)
        print(f"Mode: HYBRID ({oneply_pct}% 1-ply + {mcts_pct}% MCTS)")
        print(f"  MCTS simulations/move: {config.hybrid_mcts_simulations}")
        print(f"  MCTS noise: {noise_info()}")
    elif config.use_mcts_rust:
        print(f"Mode: Rust MCTS (fast)")
        print(f"  Simulations/move: {config.mcts_simulations}")
        print(f"  MCTS noise: {noise_info()}")
    elif config.use_mcts:
        print(f"Mode: Python MCTS")
        print(f"  Simulations/move: {config.mcts_simulations}")
        print(f"  MCTS noise: {noise_info()}")
    elif config.use_rust:
        print(f"Mode: Rust 1-ply (fastest)")
    else:
        print(f"Mode: Python batched 1-ply")
    print(f"Outcomes: Win=+1.0, Loss=-{config.loss_penalty}, Draw=-{config.draw_penalty}")
    if config.combo_bonus_enabled:
        print(f"Combo bonus: +{config.combo_bonus_per_attack}/attack during Royal Decree (max: +{config.combo_bonus_max})")
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
