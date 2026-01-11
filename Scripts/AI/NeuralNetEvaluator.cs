using System;
using System.IO;
using Godot;
using Exchange.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Exchange.AI;

/// <summary>
/// Neural network-based position evaluator using ONNX Runtime.
/// Replaces the hand-tuned Evaluate() function with a trained model.
///
/// The network outputs a value in [-1, 1]:
/// - Positive = good for the side to move
/// - Negative = bad for the side to move
/// - Magnitude indicates confidence
///
/// To use: Set UseNeuralNet = true on AISearchEngine after loading a model.
/// </summary>
public class NeuralNetEvaluator : IDisposable
{
    private InferenceSession? _session;
    private bool _isLoaded;
    private string? _modelPath;

    // Input tensor dimensions
    private const int InputChannels = 27;
    private const int BoardSize = 8;

    // Scaling factor to convert [-1,1] to centipawn-like values
    private const float ScoreScale = 10000f;

    /// <summary>
    /// Whether a model is loaded and ready for inference.
    /// </summary>
    public bool IsLoaded => _isLoaded && _session != null;

    /// <summary>
    /// Load an ONNX model from the given path.
    /// </summary>
    /// <param name="modelPath">Path to .onnx file</param>
    /// <returns>True if loaded successfully</returns>
    public bool LoadModel(string modelPath)
    {
        try
        {
            Dispose(); // Clean up any existing session

            if (!File.Exists(modelPath))
            {
                GameLogger.Error("NeuralNet", $"Model file not found: {modelPath}");
                return false;
            }

            // Create session options for optimal performance
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            };

            // Use CoreML on Mac for hardware acceleration
            #if GODOT_MACOS
            try
            {
                options.AppendExecutionProvider_CoreML();
                GameLogger.Debug("NeuralNet", "Using CoreML execution provider");
            }
            catch
            {
                GameLogger.Debug("NeuralNet", "CoreML not available, using CPU");
            }
            #endif

            _session = new InferenceSession(modelPath, options);
            _modelPath = modelPath;
            _isLoaded = true;

            GameLogger.Info("NeuralNet", $"Loaded model: {modelPath}");
            return true;
        }
        catch (Exception ex)
        {
            GameLogger.Error("NeuralNet", $"Failed to load model: {ex.Message}");
            _isLoaded = false;
            return false;
        }
    }

    /// <summary>
    /// Try to load model from default location in project.
    /// Looks for models/value_network.onnx relative to the executable.
    /// </summary>
    public bool TryLoadDefaultModel()
    {
        string[] searchPaths =
        {
            "res://models/value_network.onnx",
            "models/value_network.onnx",
            "../Exchange.Training/models/value_network.onnx",
        };

        foreach (var path in searchPaths)
        {
            var globalPath = path.StartsWith("res://")
                ? ProjectSettings.GlobalizePath(path)
                : path;

            if (File.Exists(globalPath))
            {
                return LoadModel(globalPath);
            }
        }

        GameLogger.Warning("NeuralNet", "No default model found. Train one with Exchange.Training!");
        return false;
    }

    /// <summary>
    /// Evaluate a board position using the neural network.
    /// Returns score from the perspective of sideToMove.
    /// </summary>
    /// <param name="state">The simulated board state</param>
    /// <param name="sideToMove">Which team is to move</param>
    /// <returns>Score scaled to centipawn-like values</returns>
    public int Evaluate(SimulatedBoardState state, Team sideToMove)
    {
        if (!IsLoaded)
        {
            GameLogger.Warning("NeuralNet", "Model not loaded, returning 0");
            return 0;
        }

        try
        {
            // Convert state to tensor
            var inputTensor = StateToTensor(state, sideToMove);

            // Run inference
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
            using var results = _session!.Run(inputs);

            // Get output value
            var outputTensor = results[0].AsTensor<float>();
            float rawValue = outputTensor.GetValue(0);

            // Scale to centipawn-like value
            int score = (int)(rawValue * ScoreScale);

            return score;
        }
        catch (Exception ex)
        {
            GameLogger.Error("NeuralNet", $"Inference error: {ex.Message}");
            return 0;
        }
    }

    /// <summary>
    /// Convert SimulatedBoardState to input tensor for the network.
    /// Must match the Python encoding in game_state.py exactly!
    /// </summary>
    private DenseTensor<float> StateToTensor(SimulatedBoardState state, Team sideToMove)
    {
        // Shape: [1, 27, 8, 8] (batch=1, channels=27, height=8, width=8)
        var tensor = new DenseTensor<float>(new[] { 1, InputChannels, BoardSize, BoardSize });

        // Channel layout (must match Python!):
        // 0-5: Player piece positions (King, Queen, Rook, Bishop, Knight, Pawn)
        // 6-11: Enemy piece positions
        // 12-17: HP planes (normalized, one per piece type)
        // 18-23: Cooldown planes (normalized)
        // 24: Royal Decree active for side to move
        // 25: Interpose positions
        // 26: Side to move (1 = player, 0 = enemy)

        foreach (var piece in state.AllPieces)
        {
            if (!piece.IsAlive) continue;

            int x = piece.Position.X;
            int y = piece.Position.Y;
            int pt = (int)piece.PieceType;
            int teamOffset = piece.Team == Team.Player ? 0 : 6;

            // Piece position plane
            tensor[0, teamOffset + pt, y, x] = 1.0f;

            // HP plane (normalized)
            float hpRatio = (float)piece.CurrentHp / piece.MaxHp;
            tensor[0, 12 + pt, y, x] = hpRatio;

            // Cooldown plane - would need to be added to SimulatedPiece
            // For now, assume no cooldown tracking in simulation
            // tensor[0, 18 + pt, y, x] = cooldownRatio;
        }

        // Side to move plane
        if (sideToMove == Team.Player)
        {
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    tensor[0, 26, y, x] = 1.0f;
        }

        return tensor;
    }

    public void Dispose()
    {
        _session?.Dispose();
        _session = null;
        _isLoaded = false;
    }
}

/// <summary>
/// Extension methods for AISearchEngine to use neural network evaluation.
/// </summary>
public static class NeuralNetExtensions
{
    private static NeuralNetEvaluator? _evaluator;
    private static bool _useNeuralNet;

    /// <summary>
    /// Initialize neural network evaluator.
    /// Call this once at game start.
    /// </summary>
    public static bool InitializeNeuralNet(string? modelPath = null)
    {
        _evaluator = new NeuralNetEvaluator();

        if (modelPath != null)
        {
            return _evaluator.LoadModel(modelPath);
        }
        else
        {
            return _evaluator.TryLoadDefaultModel();
        }
    }

    /// <summary>
    /// Enable or disable neural network evaluation.
    /// When disabled, falls back to hand-tuned evaluation.
    /// </summary>
    public static void SetUseNeuralNet(bool use)
    {
        _useNeuralNet = use && (_evaluator?.IsLoaded ?? false);
        GameLogger.Info("NeuralNet", $"Neural net evaluation: {(_useNeuralNet ? "ENABLED" : "DISABLED")}");
    }

    /// <summary>
    /// Check if neural net is available and enabled.
    /// </summary>
    public static bool IsNeuralNetEnabled => _useNeuralNet && (_evaluator?.IsLoaded ?? false);

    /// <summary>
    /// Get the neural net evaluator instance.
    /// </summary>
    public static NeuralNetEvaluator? GetEvaluator() => _evaluator;

    /// <summary>
    /// Evaluate a position using neural net if available, otherwise return null.
    /// </summary>
    public static int? TryEvaluateWithNeuralNet(SimulatedBoardState state, Team sideToMove)
    {
        if (!_useNeuralNet || _evaluator == null || !_evaluator.IsLoaded)
            return null;

        return _evaluator.Evaluate(state, sideToMove);
    }
}
