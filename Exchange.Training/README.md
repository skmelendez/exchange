# EXCHANGE AI Training

Neural network training for the EXCHANGE chess-variant roguelike.

## Overview

This module trains a **Value Network** that evaluates board positions, replacing the hand-tuned 1200-line evaluation function in `AISearchEngine.cs`. The trained model is exported to ONNX format for inference in Godot.

## Architecture

```
Input: 27x8x8 tensor (board state)
       ├── 12 planes: piece positions (6 types x 2 teams)
       ├── 6 planes: normalized HP per piece type
       ├── 6 planes: ability cooldowns
       ├── 2 planes: special states (Royal Decree, Interpose)
       └── 1 plane: side to move

Network: ResNet-style convolutional
         ├── Input conv: 27 -> 64 channels
         ├── 4 residual blocks (3x3 conv)
         ├── Value head: conv + FC layers
         └── Output: tanh activation

Output: Single value [-1, 1]
        ├── Positive = good for side to move
        └── Negative = bad for side to move
```

## Quick Start

```bash
# 1. Create virtual environment
cd Exchange.Training
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate   # On Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Quick test (10 iterations, ~5 minutes)
python scripts/train.py --quick

# 4. Full training (100 iterations, ~2-4 hours on M4)
python scripts/train.py --preset medium --iterations 100 --games 200

# 5. Export to ONNX
python -m src.onnx_export runs/experiment/checkpoints/best.pt ../models/value_network.onnx

# 6. Copy to Godot project
mkdir -p ../models
cp runs/experiment/checkpoints/best.onnx ../models/value_network.onnx
```

## Training Pipeline

### Phase 1: Bootstrap
- Plays 1000 random games to seed initial training data
- Network learns basic patterns from random play outcomes
- Takes ~10-15 minutes on M4

### Phase 2: Self-Play
- Network plays against itself to generate training data
- Temperature-based move selection for exploration
- Temperature anneals from 1.0 -> 0.1 over training
- Each iteration:
  1. Generate N games using current network
  2. Add positions to replay buffer
  3. Train network on accumulated data
  4. Evaluate against random opponent
  5. Save checkpoints

### Phase 3: Export
- Best model exported to ONNX format
- Verified for numerical accuracy
- Performance tested for inference speed

## Configuration

```python
TrainingConfig(
    # Network
    network_preset="medium",      # tiny/small/medium/large

    # Training
    batch_size=256,
    learning_rate=1e-3,
    epochs_per_iteration=10,

    # Self-play
    games_per_iteration=200,      # Games per training iteration
    max_turns_per_game=200,       # Prevent infinite games
    temperature_start=1.0,        # High exploration initially
    temperature_end=0.1,          # Exploit more as training progresses

    # Bootstrap
    bootstrap_games=1000,         # Random games to seed training

    # Loop
    num_iterations=100,           # Total training iterations
    eval_interval=5,              # Evaluate every N iterations
    checkpoint_interval=10,       # Save every N iterations
)
```

## Network Presets

| Preset | Params | Inference | Recommended For |
|--------|--------|-----------|-----------------|
| tiny   | ~30k   | <0.5ms    | Quick testing   |
| small  | ~70k   | ~0.5ms    | Fast inference  |
| medium | ~150k  | ~1ms      | Balanced (default) |
| large  | ~400k  | ~2ms      | Maximum strength |

## Monitoring Training

TensorBoard logs are saved to `runs/experiment/logs/`:

```bash
tensorboard --logdir runs/experiment/logs
```

Metrics tracked:
- `training/loss`: MSE loss on position values
- `training/temperature`: Current exploration temperature
- `training/buffer_size`: Replay buffer size
- `eval/win_rate`: Win rate against random opponent

## Personality Training (Coming Soon)

Train specialized models with reward shaping:

```python
# Aggressive: bonus for damage dealt
# Defensive: bonus for HP preserved
# King Hunter: bonus for king attacks
# Berserker: ignore own HP
```

## Integration with Godot

1. Copy `models/value_network.onnx` to your Godot project
2. Enable neural net in `AIController.cs`:

```csharp
// Initialize neural net evaluator
NeuralNetExtensions.InitializeNeuralNet("res://models/value_network.onnx");
NeuralNetExtensions.SetUseNeuralNet(true);
```

3. The `AISearchEngine` will automatically use neural net evaluation if enabled

## File Structure

```
Exchange.Training/
├── src/
│   ├── game_state.py      # Board state representation
│   ├── value_network.py   # PyTorch network architecture
│   ├── game_simulator.py  # Python game rules (mirrors C#)
│   ├── training.py        # Self-play training loop
│   └── onnx_export.py     # ONNX export utility
├── scripts/
│   └── train.py           # Main training entry point
├── data/                  # (generated) Training data
├── models/                # (generated) Exported ONNX models
├── runs/                  # (generated) Training runs
│   └── experiment/
│       ├── checkpoints/   # PyTorch checkpoints
│       ├── logs/          # TensorBoard logs
│       └── models/        # ONNX exports
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (MPS support for M4 Mac)
- ONNX Runtime 1.15+
- ~4GB RAM for training
- ~30-60 minutes for quick test
- ~2-4 hours for full training

## Troubleshooting

**Model not loading in Godot:**
- Ensure ONNX file is in the correct location
- Check file permissions
- Verify ONNX Runtime NuGet package is installed

**Training too slow:**
- Reduce `games_per_iteration`
- Use `--preset tiny` for faster experiments
- Check that MPS (Apple GPU) is being used on Mac

**Win rate not improving:**
- Increase bootstrap games
- Train for more iterations
- Try larger network preset
