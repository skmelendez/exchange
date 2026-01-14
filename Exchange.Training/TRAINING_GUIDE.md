# EXCHANGE AI Training Guide

This guide covers how to train the EXCHANGE neural network, monitor progress, and review games.

## Table of Contents

- [Quick Start](#quick-start)
- [Setup](#setup)
- [Training Commands](#training-commands)
- [Monitoring with TensorBoard](#monitoring-with-tensorboard)
- [Game Dashboard](#game-dashboard)
- [Checkpoints & Resuming](#checkpoints--resuming)
- [Command Reference](#command-reference)
- [Recommended Workflows](#recommended-workflows)

---

## Quick Start

```bash
# 1. Activate virtual environment
cd Exchange.Training
source venv/bin/activate

# 2. Run a quick test (verifies everything works)
python scripts/train.py --quick

# 3. Launch TensorBoard to monitor (in another terminal)
tensorboard --logdir runs/experiment/logs

# 4. Launch game dashboard to watch replays (in another terminal)
python -m src.dashboard --replays runs/experiment/replays
```

---

## Setup

### Prerequisites

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build Rust simulator (optional, for 100x faster MCTS)
cd rust_simulator && ./setup.sh && cd ..
```

### Directory Structure

After training, your output directory will contain:

```
runs/experiment/
├── checkpoints/
│   ├── latest.pt      # Most recent checkpoint
│   ├── best.pt        # Best performing model (champion)
│   └── iter_0010.pt   # Periodic checkpoints
├── logs/              # TensorBoard logs
│   └── events.out.tfevents.*
├── replays/           # Game replays for review
│   ├── iter_0000/
│   │   ├── game_0001.json
│   │   └── game_0002.json
│   └── iter_0010/
└── models/
    └── value_network.onnx  # Exported model for Godot
```

---

## Training Commands

### Basic Training

```bash
# Full training with default settings (100 iterations, medium network)
python scripts/train.py

# Quick test run (tiny network, 10 iterations, 20 games each)
python scripts/train.py --quick

# Custom iteration count
python scripts/train.py --iterations 50

# Custom games per iteration
python scripts/train.py --games 100
```

### Network Size Presets

| Preset   | Parameters | Description                          |
|----------|------------|--------------------------------------|
| `tiny`   | ~50K       | Fast training, good for testing      |
| `small`  | ~200K      | Balance of speed and quality         |
| `medium` | ~21M       | Full-featured with policy head       |
| `large`  | ~50M       | Maximum capacity (slower)            |

```bash
# Use a specific network size
python scripts/train.py --preset small
python scripts/train.py --preset large
```

### MCTS (Monte Carlo Tree Search)

MCTS provides lookahead during self-play, generating higher quality training data.

```bash
# Standard Python MCTS (slower, ~6 sec/game)
python scripts/train.py --mcts --mcts-sims 50

# Rust MCTS (recommended, ~4 sec/game with 50 sims)
python scripts/train.py --mcts-rust --mcts-sims 50

# More simulations = stronger play, slower training
python scripts/train.py --mcts-rust --mcts-sims 100
python scripts/train.py --mcts-rust --mcts-sims 200
```

**MCTS Simulation Guidelines:**

| Simulations | Quality | Speed       | Use Case                    |
|-------------|---------|-------------|-----------------------------|
| 25          | Basic   | ~2s/game    | Quick iteration, debugging  |
| 50          | Good    | ~4s/game    | Standard training           |
| 100         | Strong  | ~8s/game    | Quality training            |
| 200         | Expert  | ~15s/game   | Final polish, evaluation    |

### Learning Rate & Batch Size

```bash
# Adjust learning rate (default: 0.001)
python scripts/train.py --lr 0.0005

# Adjust batch size (default: 256)
python scripts/train.py --batch-size 512
```

### Bootstrap Phase

The bootstrap phase generates initial training data with random play before the main training loop.

```bash
# Default: 1000 bootstrap games
python scripts/train.py --bootstrap-games 1000

# Faster startup with fewer games
python scripts/train.py --bootstrap-games 500

# More data for better initial training
python scripts/train.py --bootstrap-games 2000
```

### Output Directory

```bash
# Save to custom directory
python scripts/train.py --output-dir runs/my_experiment

# Different experiment names
python scripts/train.py --output-dir runs/mcts_v1
python scripts/train.py --output-dir runs/large_network
```

### Parallel Workers

```bash
# Auto-detect CPU cores (default)
python scripts/train.py --workers 0

# Specify worker count
python scripts/train.py --workers 8
```

### ONNX Export

```bash
# Export best model to ONNX after training completes
python scripts/train.py --export-onnx

# Manual export of specific checkpoint
python -m src.onnx_export runs/experiment/checkpoints/best.pt models/value_network.onnx
```

---

## Monitoring with TensorBoard

TensorBoard provides real-time training visualization.

### Launch TensorBoard

```bash
# Standard launch
tensorboard --logdir runs/experiment/logs

# Specify port
tensorboard --logdir runs/experiment/logs --port 6006

# Monitor multiple experiments
tensorboard --logdir runs/
```

Then open http://localhost:6006 in your browser.

### Available Metrics

| Metric | Description |
|--------|-------------|
| `Loss/train` | Training loss per iteration |
| `Loss/value` | Value head loss |
| `Loss/policy` | Policy head loss (MCTS only) |
| `Win_Rate/vs_random` | Win rate against random opponent |
| `Win_Rate/vs_champion` | Win rate against previous best model |
| `Games/white_wins` | White win percentage |
| `Games/black_wins` | Black win percentage |
| `Games/draws` | Draw percentage |
| `Temperature` | Current exploration temperature |
| `Abilities/rd_activations` | Royal Decree activations |
| `Abilities/rd_combo_attacks` | Attacks during Royal Decree |
| `Abilities/interpose_triggers` | Reactive Interpose triggers |
| `Abilities/consecration_uses` | Bishop heal uses |
| `Abilities/pawn_promotions` | Pawns promoted to Queen |
| `Rewards/king_hunt_bonus` | King proximity/attack bonuses |
| `Rewards/shuffle_penalty` | Accumulated no-damage penalties |

---

## Game Dashboard

The web dashboard lets you visually review training games.

### Launch Dashboard

```bash
# Standard launch (default replays directory)
python -m src.dashboard --replays runs/experiment/replays

# Specify port
python -m src.dashboard --replays runs/experiment/replays --port 8080

# Access from other devices on network
python -m src.dashboard --replays runs/experiment/replays --host 0.0.0.0
```

Then open http://localhost:5000 in your browser.

### Dashboard Features

- **Browse saved replays** by iteration
- **Step through moves** with Next/Prev buttons
- **Scrubber bar** to jump to any point in the game
- **Autoplay** with adjustable speed
- **HP bars** showing piece health
- **Movement/attack lines** visualizing each move
- **Filter games** by outcome (White wins, Black wins, Draws)
- **Watch new game** - play a random game live

### Command-Line Game Viewer

```bash
# Watch a random game in terminal
python -m src.game_viewer --watch

# Watch AI play (uses trained model)
python -m src.game_viewer --watch --model runs/experiment/checkpoints/best.pt

# Replay a saved game
python -m src.game_viewer --replay runs/experiment/replays/iter_0000/game_0001.json

# Record random games
python -m src.game_viewer --record 10
```

---

## Checkpoints & Resuming

### Checkpoint Types

| File | Description |
|------|-------------|
| `latest.pt` | Most recent checkpoint (saved every iteration) |
| `best.pt` | Champion model (highest win rate vs previous champion) |
| `iter_XXXX.pt` | Periodic checkpoints (every 10 iterations by default) |

### Resume Training

```bash
# Resume from latest checkpoint
python scripts/train.py --resume runs/experiment/checkpoints/latest.pt

# Resume from specific iteration
python scripts/train.py --resume runs/experiment/checkpoints/iter_0050.pt

# Resume from best model
python scripts/train.py --resume runs/experiment/checkpoints/best.pt

# Resume with different settings (e.g., continue with MCTS)
python scripts/train.py --resume runs/experiment/checkpoints/latest.pt --mcts-rust --mcts-sims 100

# Resume to different output directory
python scripts/train.py --resume runs/old_experiment/checkpoints/best.pt --output-dir runs/new_experiment
```

### What's Saved in Checkpoints

- Network weights
- Optimizer state
- Training iteration number
- Best validation metrics
- Network configuration
- Training configuration

---

## Command Reference

### `scripts/train.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` | - | Quick test mode (tiny network, 10 iter, 20 games) |
| `--preset` | `medium` | Network size: `tiny`, `small`, `medium`, `large` |
| `--iterations` | `100` | Number of training iterations |
| `--games` | `200` | Games per iteration |
| `--bootstrap-games` | `1000` | Initial random games for bootstrap |
| `--batch-size` | `256` | Training batch size |
| `--lr` | `0.001` | Learning rate |
| `--workers` | `0` | Parallel workers (0=auto) |
| `--mcts` | - | Enable Python MCTS |
| `--mcts-rust` | - | Enable Rust MCTS (faster) |
| `--mcts-sims` | `100` | MCTS simulations per move |
| `--resume` | - | Path to checkpoint to resume from |
| `--output-dir` | `runs/experiment` | Output directory |
| `--export-onnx` | - | Export to ONNX after training |
| `--use-new-rewards` | `True` | Use new reward system with ability tracking |
| `--asymmetric` | - | Enable asymmetric training (MCTS vs Network) |
| `--asymmetric-sims` | `50` | MCTS simulations for asymmetric mode |

### `src.dashboard`

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `5000` | HTTP server port |
| `--replays` | `runs/experiment/replays` | Replays directory |
| `--host` | `0.0.0.0` | Host to bind to |

### `src.game_viewer`

| Flag | Default | Description |
|------|---------|-------------|
| `--watch` | - | Watch a live game |
| `--model` | - | Model checkpoint for AI player |
| `--replay` | - | Path to replay file |
| `--record` | - | Number of random games to record |
| `--delay` | `0.3` | Delay between moves (seconds) |
| `--no-unicode` | - | Use ASCII instead of Unicode pieces |

---

## Recommended Workflows

### First Time Training

```bash
# 1. Test everything works
python scripts/train.py --quick

# 2. Check TensorBoard
tensorboard --logdir runs/experiment/logs

# 3. Run real training
python scripts/train.py --preset medium --iterations 50 --games 200

# 4. Review games in dashboard
python -m src.dashboard --replays runs/experiment/replays

# 5. Export for Godot
python -m src.onnx_export runs/experiment/checkpoints/best.pt models/value_network.onnx
```

### Training with MCTS (Recommended for Quality)

```bash
# Build Rust simulator first
cd rust_simulator && ./setup.sh && cd ..

# Train with Rust MCTS
python scripts/train.py --mcts-rust --mcts-sims 50 --iterations 100 --games 100
```

### Long Training Session

```bash
# Start training with generous settings
python scripts/train.py \
    --preset medium \
    --iterations 200 \
    --games 200 \
    --mcts-rust \
    --mcts-sims 100 \
    --output-dir runs/production_v1

# If interrupted, resume:
python scripts/train.py \
    --resume runs/production_v1/checkpoints/latest.pt \
    --mcts-rust \
    --mcts-sims 100
```

### Comparing Experiments

```bash
# Run multiple experiments with different settings
python scripts/train.py --output-dir runs/exp_no_mcts --iterations 50
python scripts/train.py --output-dir runs/exp_mcts_50 --mcts-rust --mcts-sims 50 --iterations 50
python scripts/train.py --output-dir runs/exp_mcts_100 --mcts-rust --mcts-sims 100 --iterations 50

# Compare in TensorBoard
tensorboard --logdir runs/
```

---

## Troubleshooting

### CUDA/MPS Out of Memory

```bash
# Reduce batch size
python scripts/train.py --batch-size 128

# Use smaller network
python scripts/train.py --preset small
```

### Training Too Slow

```bash
# Use Rust MCTS instead of Python MCTS
python scripts/train.py --mcts-rust --mcts-sims 50

# Reduce simulations
python scripts/train.py --mcts-rust --mcts-sims 25

# Disable MCTS (fastest, but lower quality)
python scripts/train.py
```

### Rust Simulator Not Found

```bash
# Rebuild Rust simulator
cd rust_simulator
./setup.sh
cd ..

# Verify installation
python -c "from exchange_simulator import RustSimulator; print('OK')"
```

---

## Performance Tips

1. **Use Rust MCTS** (`--mcts-rust`) - 10-50x faster than Python MCTS
2. **Start with lower simulations** (25-50) and increase as you validate
3. **Monitor TensorBoard** for early signs of training issues
4. **Review games in dashboard** to verify AI is learning reasonable strategies
5. **Use checkpoints** - training can always be resumed
6. **Export to ONNX** for deployment in Godot with fast inference
