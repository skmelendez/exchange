using Godot;
using Exchange.Core;
using static Exchange.Core.SaveManager;

namespace Exchange.Map;

/// <summary>
/// Manages the overall roguelike run state.
/// Tracks current act, map position, resources, and relics.
/// </summary>
public partial class RunManager : Node
{
    // Godot signals for simple types
    [Signal] public delegate void ActStartedEventHandler(int actNumber);
    [Signal] public delegate void ActCompletedEventHandler(int actNumber);
    [Signal] public delegate void RunCompletedEventHandler(bool victory);
    [Signal] public delegate void MapUpdatedEventHandler();

    // C# events for complex types (Godot signals don't support custom classes)
    public event Action<MapNode>? NodeSelected;
    public event Action<MapNode>? NodeCompleted;

    // Run state
    public int RunSeed { get; private set; }
    public int CurrentActNumber { get; private set; } = 0;
    public ActMap? CurrentActMap { get; private set; }
    public MapNode? CurrentNode { get; private set; }
    public bool IsRunActive { get; private set; } = false;

    // Resources
    public int Coins { get; private set; } = 0;
    public int MaxHealth { get; private set; } = 100;  // Future: King's total HP pool across matches
    public int CurrentHealth { get; private set; } = 100;

    // Relics (future)
    public List<string> Relics { get; } = new();

    // Stats
    public int TotalCombatsWon { get; private set; } = 0;
    public int TotalCoinsEarned { get; private set; } = 0;

    // Get configs from constants
    private static ActConfig[] ActConfigs => MapConstants.GetAllActConfigs();

    /// <summary>
    /// Start a new run with optional seed (0 = random)
    /// </summary>
    public void StartNewRun(int seed = 0)
    {
        RunSeed = seed == 0 ? (int)DateTime.Now.Ticks : seed;
        CurrentActNumber = 0;
        Coins = 0;
        CurrentHealth = MaxHealth;
        Relics.Clear();
        TotalCombatsWon = 0;
        TotalCoinsEarned = 0;
        IsRunActive = true;

        GameLogger.Info("Run", $"Starting new run with seed {RunSeed}");

        // Start Act 1
        StartAct(1);
    }

    /// <summary>
    /// Start a specific act
    /// </summary>
    public void StartAct(int actNumber)
    {
        if (actNumber < 1 || actNumber > ActConfigs.Length)
        {
            GameLogger.Error("Run", $"Invalid act number: {actNumber}");
            return;
        }

        CurrentActNumber = actNumber;
        var config = ActConfigs[actNumber - 1];

        // Generate map with act-specific seed
        int actSeed = RunSeed + actNumber * 1000;
        CurrentActMap = MapGenerator.Generate(config, actSeed);

        // Reset to map view (no current node = choose starting node)
        CurrentNode = null;
        CurrentActMap.SetCurrentNode(null);

        GameLogger.Info("Run", $"Started Act {actNumber} (AI Depth: {config.AiDepth})");
        EmitSignal(SignalName.ActStarted, actNumber);
        EmitSignal(SignalName.MapUpdated);
    }

    /// <summary>
    /// Regenerate current act with custom config and seed (for debug menu)
    /// </summary>
    public void RegenerateWithConfig(ActConfig customConfig, int seed)
    {
        GameLogger.Info("RunManager", $"Regenerating map - Seed: {seed}, Columns: {customConfig.Columns}, Rows: {customConfig.MaxRows}, AI: {customConfig.AiDepth}");

        CurrentActMap = MapGenerator.Generate(customConfig, seed);

        // Reset to map view
        CurrentNode = null;
        CurrentActMap.SetCurrentNode(null);

        EmitSignal(SignalName.MapUpdated);
    }

    /// <summary>
    /// Get the AI depth for current act (respects debug overrides)
    /// </summary>
    public int GetCurrentAiDepth()
    {
        // Use map's effective depth if available (includes override)
        if (CurrentActMap != null)
            return CurrentActMap.EffectiveAiDepth;

        // Fallback to config
        if (CurrentActNumber < 1 || CurrentActNumber > ActConfigs.Length)
            return 2;  // Default
        return ActConfigs[CurrentActNumber - 1].AiDepth;
    }

    /// <summary>
    /// Update AI depth without regenerating the map (for debug menu)
    /// </summary>
    public void SetAiDepth(int depth)
    {
        if (CurrentActMap == null)
        {
            GameLogger.Warning("RunManager", "Cannot set AI depth: no current map");
            return;
        }

        CurrentActMap.SetAiDepthOverride(depth);
        GameLogger.Info("RunManager", $"AI depth updated to {depth}-ply");
        EmitSignal(SignalName.MapUpdated);
    }

    /// <summary>
    /// Select a node to travel to (must be accessible)
    /// </summary>
    public bool SelectNode(MapNode node)
    {
        if (CurrentActMap == null)
        {
            GameLogger.Warning("RunManager", "SelectNode failed: CurrentActMap is null");
            return false;
        }
        if (!node.IsAccessible)
        {
            GameLogger.Warning("RunManager", $"SelectNode failed: Node {node.NodeType} at {node.Position} is NOT accessible");
            GameLogger.Debug("RunManager", $"  CurrentNode: {CurrentNode?.NodeType} at {CurrentNode?.Position}");
            GameLogger.Debug("RunManager", $"  CurrentNode outgoing: {string.Join(", ", CurrentNode?.OutgoingConnections ?? [])}");
            return false;
        }

        CurrentNode = node;
        CurrentActMap.SetCurrentNode(node);

        GameLogger.Info("Run", $"Selected node: {node}");
        NodeSelected?.Invoke(node);
        EmitSignal(SignalName.MapUpdated);

        return true;
    }

    /// <summary>
    /// Select a node by its ID
    /// </summary>
    public bool SelectNodeById(int nodeId)
    {
        var node = CurrentActMap?.GetNodeById(nodeId);
        if (node == null) return false;
        return SelectNode(node);
    }

    /// <summary>
    /// Called when current node encounter is completed
    /// </summary>
    public void CompleteCurrentNode(bool success, int coinsEarned = 0)
    {
        if (CurrentNode == null) return;

        GameLogger.Info("Run", $"Node completed: {CurrentNode.NodeType} - Success: {success}, Coins: {coinsEarned}");

        if (CurrentNode.IsCombatNode)
        {
            if (success)
            {
                TotalCombatsWon++;
                AddCoins(coinsEarned);
                CurrentNode.IsCompleted = true;
            }
            else
            {
                // Combat loss = run over
                EndRun(false);
                return;
            }
        }
        else
        {
            // Non-combat nodes always succeed
            AddCoins(coinsEarned);
            CurrentNode.IsCompleted = true;
        }

        NodeCompleted?.Invoke(CurrentNode);

        // Check if this was the boss
        if (CurrentNode.NodeType == MapNodeType.Boss)
        {
            CompleteAct();
            return;
        }

        // Update map accessibility for next selection
        EmitSignal(SignalName.MapUpdated);
    }

    /// <summary>
    /// Called when an act boss is defeated
    /// </summary>
    private void CompleteAct()
    {
        GameLogger.Info("Run", $"Act {CurrentActNumber} completed!");
        EmitSignal(SignalName.ActCompleted, CurrentActNumber);

        if (CurrentActNumber >= ActConfigs.Length)
        {
            // All acts complete = victory!
            EndRun(true);
        }
        else
        {
            // Start next act
            StartAct(CurrentActNumber + 1);
        }
    }

    /// <summary>
    /// End the run (victory or defeat)
    /// </summary>
    public void EndRun(bool victory)
    {
        IsRunActive = false;

        GameLogger.Info("Run", $"Run ended - {(victory ? "VICTORY!" : "DEFEAT")}");
        GameLogger.Info("Run", $"Stats: Combats Won: {TotalCombatsWon}, Coins Earned: {TotalCoinsEarned}");

        EmitSignal(SignalName.RunCompleted, victory);
    }

    /// <summary>
    /// Add coins to player
    /// </summary>
    public void AddCoins(int amount)
    {
        if (amount <= 0) return;
        Coins += amount;
        TotalCoinsEarned += amount;
        GameLogger.Info("Run", $"+{amount} coins (Total: {Coins})");
    }

    /// <summary>
    /// Spend coins (returns false if not enough)
    /// </summary>
    public bool SpendCoins(int amount)
    {
        if (amount > Coins) return false;
        Coins -= amount;
        GameLogger.Info("Run", $"-{amount} coins (Total: {Coins})");
        return true;
    }

    /// <summary>
    /// Get list of accessible nodes for UI
    /// </summary>
    public List<MapNode> GetAccessibleNodes()
    {
        return CurrentActMap?.GetAccessibleNodes() ?? new List<MapNode>();
    }

    /// <summary>
    /// Check if we're on the map screen (no active node encounter)
    /// </summary>
    public bool IsOnMapScreen => IsRunActive && CurrentNode != null &&
        (CurrentNode.IsVisited && CurrentNode.OutgoingConnections.Count > 0);

    /// <summary>
    /// DEBUG: Skip to specific act
    /// </summary>
    public void DebugSkipToAct(int actNumber)
    {
        if (actNumber < 1 || actNumber > ActConfigs.Length)
        {
            GameLogger.Warning("RunManager", $"Invalid act number: {actNumber}");
            return;
        }

        GameLogger.Info("RunManager", $"[DEBUG] Skipping to Act {actNumber}");
        StartAct(actNumber);
    }

    #region Save/Load

    /// <summary>
    /// Create save data from current run state.
    /// </summary>
    public SaveData ToSaveData()
    {
        var saveData = new SaveData
        {
            RunSeed = RunSeed,
            CurrentActNumber = CurrentActNumber,
            Coins = Coins,
            CurrentHealth = CurrentHealth,
            MaxHealth = MaxHealth,
            Relics = new List<string>(Relics),
            TotalCombatsWon = TotalCombatsWon,
            TotalCoinsEarned = TotalCoinsEarned,
            CurrentNodeId = CurrentNode?.Id,
            VisitedNodeIds = CurrentActMap?.Nodes
                .Where(n => n.IsVisited)
                .Select(n => n.Id)
                .ToList() ?? []
        };

        GameLogger.Debug("RunManager", $"Created save data: Act {saveData.CurrentActNumber}, Coins {saveData.Coins}");
        return saveData;
    }

    /// <summary>
    /// Restore run state from save data.
    /// </summary>
    public void LoadFromSaveData(SaveData saveData)
    {
        GameLogger.Info("RunManager", $"Loading save data: Act {saveData.CurrentActNumber}");

        // Restore basic state
        RunSeed = saveData.RunSeed;
        CurrentActNumber = 0; // Will be set by StartAct
        Coins = saveData.Coins;
        CurrentHealth = saveData.CurrentHealth;
        MaxHealth = saveData.MaxHealth;
        Relics.Clear();
        Relics.AddRange(saveData.Relics);
        TotalCombatsWon = saveData.TotalCombatsWon;
        TotalCoinsEarned = saveData.TotalCoinsEarned;
        IsRunActive = true;

        // Regenerate the act map with the same seed
        StartAct(saveData.CurrentActNumber);

        // Restore visited nodes
        if (CurrentActMap != null && saveData.VisitedNodeIds.Count > 0)
        {
            foreach (var nodeId in saveData.VisitedNodeIds)
            {
                var node = CurrentActMap.GetNodeById(nodeId);
                if (node != null)
                {
                    node.IsVisited = true;
                }
            }

            // Restore current node
            if (saveData.CurrentNodeId.HasValue)
            {
                var currentNode = CurrentActMap.GetNodeById(saveData.CurrentNodeId.Value);
                if (currentNode != null)
                {
                    CurrentNode = currentNode;
                    CurrentActMap.SetCurrentNode(currentNode);
                }
            }
        }

        GameLogger.Info("RunManager", $"Save loaded: Act {CurrentActNumber}, Coins {Coins}, Visited nodes: {saveData.VisitedNodeIds.Count}");
        EmitSignal(SignalName.MapUpdated);
    }

    #endregion
}
