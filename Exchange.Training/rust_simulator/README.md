# EXCHANGE Rust Simulator

High-performance game simulator for EXCHANGE training, written in Rust with Python bindings.

## Performance

Expected speedup: **50-100x** compared to Python implementation, making MCTS training practical.

| Metric | Python | Rust |
|--------|--------|------|
| Moves/sec | ~50k | ~3-5M |
| Clone state | ~50μs | ~1μs |
| MCTS 100 sims | ~3 min/game | ~2-5 sec/game |

## Quick Start

```bash
# Install and build
./setup.sh

# Test it works
python -c "from exchange_simulator import RustSimulator; s = RustSimulator(); print(f'Legal moves: {s.num_moves()}')"
```

## Manual Build

If the setup script fails:

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install maturin
pip install maturin

# Build and install the extension
maturin develop --release
```

## Usage in Python

```python
from exchange_simulator import RustSimulator

# Create simulator
sim = RustSimulator()

# Generate legal moves
moves = sim.generate_moves()
print(f"Legal moves: {len(moves)}")

# Make a move
damage = sim.make_move(moves[0])

# Clone for MCTS tree exploration
sim_copy = sim.clone_sim()

# Check game state
print(f"Terminal: {sim.is_terminal}, Winner: {sim.winner}")
```

## Integration with Training

The Rust simulator is designed as a drop-in replacement. Use `src/rust_integration.py`:

```python
from src.rust_integration import get_simulator, is_rust_available

if is_rust_available():
    sim = get_simulator(use_rust=True)
else:
    sim = get_simulator(use_rust=False)  # Falls back to Python
```

## API Reference

### RustSimulator

| Method | Description |
|--------|-------------|
| `reset()` | Reset to initial position |
| `set_seed(seed)` | Set RNG seed for reproducibility |
| `clone_sim()` | Clone simulator (fast, for MCTS) |
| `generate_moves()` | Get all legal moves |
| `make_move(move)` | Execute move, returns damage dealt |
| `make_move_by_index(idx)` | Execute move by index (faster) |
| `get_pieces()` | Get all pieces |
| `get_piece_at(x, y)` | Get piece at position |
| `num_moves()` | Count legal moves (fast) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `side_to_move` | int | 0=Player, 1=Enemy |
| `is_terminal` | bool | Game has ended |
| `winner` | int? | 0=Player, 1=Enemy, None=draw |
| `is_draw` | bool | Game ended in draw |
| `turn_number` | int | Current turn |

## Development

```bash
# Build in debug mode (faster compile, slower runtime)
maturin develop

# Build in release mode (slower compile, faster runtime)
maturin develop --release

# Run Rust tests
cargo test

# Check without building
cargo check
```
