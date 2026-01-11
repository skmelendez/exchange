# AI Training Guide

How to train and deploy neural network AI models for EXCHANGE.

---

## Quick Reference

```bash
# Start training (run before bed!)
cd Exchange.Training
source venv/bin/activate
python scripts/train.py --preset medium --iterations 100 --games 200

# Resume interrupted training
python scripts/train.py --resume runs/experiment/checkpoints/latest.pt

# Export trained model to game
python -m src.onnx_export runs/experiment/checkpoints/best.pt ../models/value_network.onnx
```

---

## Initial Setup (One Time)

```bash
# 1. Navigate to training directory
cd Exchange.Training

# 2. Create Python virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify PyTorch sees your M4 GPU
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
# Should print: MPS available: True
```

---

## Training Commands

### Quick Test (~5 minutes)
Verify everything works before a long run:
```bash
python scripts/train.py --quick
```

### Standard Training (~2-4 hours)
Good balance of quality and time:
```bash
python scripts/train.py --preset medium --iterations 100 --games 200
```

### Overnight Training (~8-12 hours)
Maximum strength, run before bed:
```bash
python scripts/train.py --preset large --iterations 200 --games 300
```

### Resume Interrupted Training
If training was stopped (Ctrl+C, sleep, etc.):
```bash
python scripts/train.py --resume runs/experiment/checkpoints/latest.pt
```

### Custom Output Directory
Keep multiple experiments organized:
```bash
python scripts/train.py --output-dir runs/aggressive_v1
python scripts/train.py --output-dir runs/defensive_v1
```

---

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | medium | Network size: tiny, small, medium, large |
| `--iterations` | 100 | Training iterations (more = stronger) |
| `--games` | 200 | Games per iteration (more = better data) |
| `--bootstrap-games` | 1000 | Initial random games to seed learning |
| `--batch-size` | 256 | Training batch size |
| `--lr` | 0.001 | Learning rate |
| `--workers` | auto | Parallel game workers (0 = all cores) |
| `--resume` | none | Checkpoint path to resume from |
| `--output-dir` | runs/experiment | Where to save everything |
| `--export-onnx` | false | Auto-export after training |

### Network Presets

| Preset | Parameters | Speed | Strength |
|--------|------------|-------|----------|
| tiny | ~30k | <0.5ms | Testing only |
| small | ~70k | ~0.5ms | Fast, decent |
| **medium** | ~150k | ~1ms | **Recommended** |
| large | ~400k | ~2ms | Maximum |

---

## Monitoring Training

### TensorBoard (Real-time graphs)
In a separate terminal:
```bash
cd Exchange.Training
source venv/bin/activate
tensorboard --logdir runs/experiment/logs
```
Then open http://localhost:6006 in browser.

**Metrics to watch:**
- `training/loss` → Should decrease over time
- `eval/win_rate` → Should increase (target: 80%+ vs random)
- `training/temperature` → Decreases from 1.0 to 0.1

### Console Output
Training prints progress like:
```
Iteration 45/100 | Loss: 0.0234 | Temp: 0.55 | Buffer: 45,000 | Time: 42.3s
  Eval: Win rate 72.0% (W:36 L:12 D:2)
```

---

## Exporting to Game

### Export Best Model
```bash
# Make sure venv is active
source venv/bin/activate

# Export best checkpoint to ONNX
python -m src.onnx_export runs/experiment/checkpoints/best.pt ../models/value_network.onnx
```

### Export Specific Checkpoint
```bash
# Export a specific iteration
python -m src.onnx_export runs/experiment/checkpoints/iter_50.pt ../models/value_network_v1.onnx
```

### Verify Export
The export script automatically verifies:
- ONNX model structure is valid
- Output matches PyTorch within tolerance
- Inference performance (~1ms target)

---

## Using Trained Model in Game

### 1. Copy Model File
The export command places it in `models/value_network.onnx`.

### 2. Enable in Code
In `AIController.cs` or game initialization:
```csharp
// Load the trained model
NeuralNetExtensions.InitializeNeuralNet("res://models/value_network.onnx");

// Enable neural net evaluation
NeuralNetExtensions.SetUseNeuralNet(true);
```

### 3. Toggle During Development
```csharp
// Switch between neural net and hand-tuned for comparison
NeuralNetExtensions.SetUseNeuralNet(useNeuralNet);

// Check if neural net is active
bool isNN = NeuralNetExtensions.IsNeuralNetEnabled;
```

---

## Overnight Training Workflow

### Before Bed
```bash
cd Exchange.Training
source venv/bin/activate

# Start a fresh long training run
python scripts/train.py --preset large --iterations 200 --games 300 --output-dir runs/overnight_$(date +%Y%m%d)

# OR resume an existing run
python scripts/train.py --resume runs/experiment/checkpoints/latest.pt --iterations 200
```

### In the Morning
```bash
# Check final results
cat runs/overnight_*/checkpoints/training_log.txt

# Export the best model
python -m src.onnx_export runs/overnight_*/checkpoints/best.pt ../models/value_network.onnx

# Test in game!
```

---

## Training Multiple Personalities

### Create Different Reward Shaping (Future)
```bash
# Aggressive AI (values damage over safety)
python scripts/train.py --output-dir runs/aggressive --reward-aggression 1.5

# Defensive AI (values HP preservation)
python scripts/train.py --output-dir runs/defensive --reward-defense 1.5

# King Hunter (obsessed with king damage)
python scripts/train.py --output-dir runs/king_hunter --reward-king-attack 2.0
```

### Export Multiple Models
```bash
python -m src.onnx_export runs/aggressive/checkpoints/best.pt ../models/ai_aggressive.onnx
python -m src.onnx_export runs/defensive/checkpoints/best.pt ../models/ai_defensive.onnx
python -m src.onnx_export runs/king_hunter/checkpoints/best.pt ../models/ai_king_hunter.onnx
```

---

## Troubleshooting

### "MPS not available"
Your Mac GPU isn't being detected:
```bash
# Update PyTorch
pip install --upgrade torch

# Verify macOS version (needs 12.3+)
sw_vers
```

### Training is slow
```bash
# Reduce games per iteration
python scripts/train.py --games 100

# Use smaller network
python scripts/train.py --preset small

# Check CPU usage - should be near 100% on all cores
top -l 1 | head -20
```

### Out of memory
```bash
# Reduce batch size
python scripts/train.py --batch-size 128

# Use smaller network
python scripts/train.py --preset small
```

### Model not loading in Godot
1. Check file exists: `ls -la models/value_network.onnx`
2. Verify ONNX package in .csproj
3. Check Godot console for error messages
4. Try absolute path: `NeuralNetExtensions.InitializeNeuralNet("/full/path/to/model.onnx")`

### Resume not working
```bash
# List available checkpoints
ls -la runs/experiment/checkpoints/

# Use specific checkpoint
python scripts/train.py --resume runs/experiment/checkpoints/iter_50.pt
```

---

## File Locations

```
Exchange.Training/
├── runs/
│   └── experiment/           # Default output
│       ├── checkpoints/
│       │   ├── latest.pt     # Most recent (for resume)
│       │   ├── best.pt       # Highest eval score
│       │   └── iter_*.pt     # Periodic saves
│       ├── logs/             # TensorBoard logs
│       └── models/           # Exported ONNX
├── venv/                     # Python environment
└── src/                      # Training code

models/                       # Game-ready ONNX models
└── value_network.onnx        # Current active model
```

---

## Recommended Training Schedule

| Day | Task | Command |
|-----|------|---------|
| 1 | Quick test | `--quick` |
| 1 | First real training | `--iterations 50` |
| 2 | Overnight run | `--iterations 100 --preset large` |
| 3+ | Continue improving | `--resume ... --iterations 200` |

Each training run builds on previous knowledge through the replay buffer. More iterations = stronger AI!

---

## Tips for Best Results

1. **Start small, verify, then go big** - Quick test first!
2. **Use TensorBoard** - Watch for loss decreasing, win rate increasing
3. **Don't interrupt early** - Let at least 20 iterations complete
4. **Save experiments** - Use `--output-dir runs/experiment_name`
5. **Compare models** - Test new exports against old ones in-game
6. **Train overnight** - Your M4 will thank you in the morning 🌙
