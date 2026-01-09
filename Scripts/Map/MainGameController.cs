using Godot;
using Exchange.Core;
using Exchange.Controllers;
using Exchange.UI;

namespace Exchange.Map;

/// <summary>
/// Top-level game controller that manages the flow between:
/// - Map screen (node selection)
/// - Combat encounters
/// - Shop/Event/Treasure screens (future)
///
/// Created and managed by GameManager. Does not auto-start.
/// </summary>
public partial class MainGameController : Node2D
{
    /// <summary>Emitted when player requests save and quit to menu.</summary>
    [Signal] public delegate void SaveAndQuitRequestedEventHandler();

    /// <summary>Emitted when game ends (victory or defeat).</summary>
    [Signal] public delegate void GameEndedEventHandler(bool victory);

    // Sub-controllers
    private RunManager _runManager = null!;
    private MapUI _mapUI = null!;
    private GameController? _combatController;

    // UI Containers
    private Control _mapContainer = null!;
    private Node2D _combatContainer = null!;
    private CanvasLayer _overlayUI = null!;

    // Menus
    private PauseMenu _pauseMenu = null!;
    private DebugMenu _debugMenu = null!;

    // State
    private enum GameScreen { Map, Combat, Shop, Event, Treasure }
    private GameScreen _currentScreen = GameScreen.Map;
    private bool _isInitialized;

    public override void _Ready()
    {
        // Process even when paused (for ESC menu handling)
        ProcessMode = ProcessModeEnum.Always;

        // Don't auto-start - wait for Initialize calls from GameManager
        GameLogger.Debug("MainGameController", "_Ready called - waiting for initialization");
    }

    /// <summary>
    /// Initialize for a new game (no save data).
    /// </summary>
    public void InitializeNewGame()
    {
        GameLogger.Info("MainGameController", "Initializing new game");
        SetupAll();
        _runManager.StartNewRun();
        ShowMapScreen();
        AutoSave(); // Save initial state
    }

    /// <summary>
    /// Initialize from saved game data.
    /// </summary>
    public void InitializeFromSave(SaveManager.SaveData saveData)
    {
        GameLogger.Info("MainGameController", $"Initializing from save - Act {saveData.CurrentActNumber}");
        SetupAll();
        _runManager.LoadFromSaveData(saveData);

        if (saveData.GameScreen == "Combat" && saveData.CombatState != null)
        {
            GameLogger.Info("MainGameController", "Restoring mid-combat save");
            // TODO: Implement combat restoration
            // For now, just show map (player will need to re-enter combat node)
            ShowMapScreen();
        }
        else
        {
            ShowMapScreen();
        }
    }

    private void SetupAll()
    {
        if (_isInitialized) return;

        SetupContainers();
        SetupRunManager();
        SetupMapUI();
        SetupOverlayUI();
        SetupMenus();

        _isInitialized = true;
        GameLogger.Debug("MainGameController", "Setup complete");
    }

    private void SetupContainers()
    {
        // Map container (full screen)
        _mapContainer = new Control
        {
            Name = "MapContainer",
            AnchorsPreset = (int)Control.LayoutPreset.FullRect
        };
        AddChild(_mapContainer);

        // Combat container (holds GameController when in combat)
        _combatContainer = new Node2D
        {
            Name = "CombatContainer",
            Visible = false
        };
        AddChild(_combatContainer);

        GameLogger.Debug("MainGameController", "Containers setup");
    }

    private void SetupRunManager()
    {
        _runManager = new RunManager();
        AddChild(_runManager);

        _runManager.NodeSelected += OnNodeSelected;
        _runManager.NodeCompleted += OnNodeCompleted;
        _runManager.ActCompleted += (actNum) => OnActCompleted(actNum);
        _runManager.RunCompleted += (victory) => OnRunCompleted(victory);

        GameLogger.Debug("MainGameController", "RunManager setup");
    }

    private void SetupMapUI()
    {
        _mapUI = new MapUI
        {
            Name = "MapUI",
            AnchorsPreset = (int)Control.LayoutPreset.FullRect
        };
        _mapUI.Initialize(_runManager);
        _mapUI.NodeClicked += OnMapNodeClicked;
        _mapContainer.AddChild(_mapUI);

        GameLogger.Debug("MainGameController", "MapUI setup");
    }

    private void SetupOverlayUI()
    {
        _overlayUI = new CanvasLayer
        {
            Name = "OverlayUI",
            Layer = 10
        };
        AddChild(_overlayUI);

        GameLogger.Debug("MainGameController", "OverlayUI setup");
    }

    private void SetupMenus()
    {
        // Pause menu
        _pauseMenu = new PauseMenu { Name = "PauseMenu" };
        _pauseMenu.ResumeRequested += OnPauseMenuResume;
        _pauseMenu.DebugMenuRequested += OnDebugMenuRequested;
        _pauseMenu.SaveAndQuitRequested += OnSaveAndQuitRequested;
        AddChild(_pauseMenu);

        // Debug menu
        _debugMenu = new DebugMenu { Name = "DebugMenu" };
        _debugMenu.Initialize(_runManager);
        _debugMenu.CloseRequested += OnDebugMenuClosed;
        _debugMenu.AutoWinRequested += OnAutoWinRequested;
        _debugMenu.RegenerateMapRequested += OnRegenerateMapRequested;
        _debugMenu.SkipToActRequested += OnSkipToActRequested;
        AddChild(_debugMenu);

        GameLogger.Debug("MainGameController", "Menus setup");
    }

    /// <summary>
    /// Handle ESC key - called by GameManager.
    /// </summary>
    public void HandleEscapeKey()
    {
        GameLogger.Debug("MainGameController", $"ESC pressed - Screen: {_currentScreen}");

        // Handle based on menu state
        if (_debugMenu.IsVisible)
        {
            GameLogger.Debug("MainGameController", "Closing debug menu");
            _debugMenu.Hide();
            _pauseMenu.Show();
        }
        else if (_pauseMenu.IsVisible)
        {
            GameLogger.Debug("MainGameController", "Closing pause menu");
            _pauseMenu.Hide();
        }
        else
        {
            // No menu open - show pause menu (works from map OR combat)
            GameLogger.Debug("MainGameController", "Showing pause menu");
            _pauseMenu.Show();
        }
    }

    /// <summary>
    /// Create save data from current game state.
    /// </summary>
    public SaveManager.SaveData? CreateSaveData()
    {
        if (_runManager == null) return null;

        var saveData = _runManager.ToSaveData();
        saveData.GameScreen = _currentScreen.ToString();

        // If in combat, save combat state
        if (_currentScreen == GameScreen.Combat && _combatController != null)
        {
            // TODO: Implement combat state serialization
            // saveData.CombatState = _combatController.ToSaveData();
        }

        GameLogger.Info("MainGameController", $"Created save data - Act {saveData.CurrentActNumber}, Screen: {saveData.GameScreen}");
        return saveData;
    }

    /// <summary>
    /// Auto-save after node completion.
    /// </summary>
    private void AutoSave()
    {
        if (_currentScreen != GameScreen.Combat)
        {
            var saveData = CreateSaveData();
            if (saveData != null)
            {
                SaveManager.Save(saveData);
                GameLogger.Debug("MainGameController", "Auto-save completed");
            }
        }
    }

    #region Menu Event Handlers

    private void OnPauseMenuResume()
    {
        GameLogger.Debug("MainGameController", "Resume from pause");
    }

    private void OnDebugMenuRequested()
    {
        GameLogger.Debug("MainGameController", "Opening debug menu");
        _pauseMenu.Hide();
        _debugMenu.Show();
    }

    private void OnSaveAndQuitRequested()
    {
        GameLogger.Info("MainGameController", "Save & Quit requested");
        EmitSignal(SignalName.SaveAndQuitRequested);
    }

    private void OnDebugMenuClosed()
    {
        GameLogger.Debug("MainGameController", "Debug menu closed");
        GetTree().Paused = false;
    }

    private void OnAutoWinRequested()
    {
        GameLogger.Info("MainGameController", "Auto-win requested");

        if (_currentScreen == GameScreen.Combat && _combatController != null)
        {
            _combatController.ForceAutoWin();
            _debugMenu.Hide();
            GetTree().Paused = false;
        }
        else
        {
            GameLogger.Warning("MainGameController", "Not in combat - cannot auto-win");
        }
    }

    private void OnRegenerateMapRequested(int seed)
    {
        var (columns, rows, aiDepth) = _debugMenu.GetMapSettings();

        GameLogger.Info("MainGameController", $"Regenerate map - Seed: {seed}, Columns: {columns}, Rows: {rows}, AI: {aiDepth}");

        var customConfig = new ActConfig
        {
            ActNumber = _runManager.CurrentActNumber,
            Columns = columns,
            MaxRows = rows,
            MinStartingPaths = 2,
            MaxStartingPaths = 3,
            EliteColumn = columns / 2,
            AiDepth = aiDepth,
            CombatWeight = 0.5f,
            EventWeight = 0.2f,
            TreasureWeight = 0.15f,
            ShopWeight = 0.15f
        };

        _runManager.RegenerateWithConfig(customConfig, seed);
        _debugMenu.Hide();
        GetTree().Paused = false;
        ShowMapScreen();
    }

    private void OnSkipToActRequested(int actNumber)
    {
        GameLogger.Info("MainGameController", $"Skip to Act {actNumber}");
        _runManager.DebugSkipToAct(actNumber);
        _debugMenu.Hide();
        GetTree().Paused = false;
        ShowMapScreen();
    }

    #endregion

    #region Map Events

    private void OnMapNodeClicked(int nodeId)
    {
        GameLogger.Debug("MainGameController", $"Map node clicked: {nodeId}");

        if (_runManager.SelectNodeById(nodeId))
        {
            var node = _runManager.CurrentNode;
            if (node != null)
            {
                EnterNode(node);
            }
        }
        else
        {
            GameLogger.Debug("MainGameController", "Node not accessible");
        }
    }

    private void OnNodeSelected(MapNode node)
    {
        GameLogger.Info("MainGameController", $"Node selected: {node.NodeType} at {node.Position}");
    }

    private void EnterNode(MapNode node)
    {
        GameLogger.Info("MainGameController", $"Entering node: {node.NodeType}");

        switch (node.NodeType)
        {
            case MapNodeType.Combat:
            case MapNodeType.Elite:
            case MapNodeType.Boss:
                StartCombat(node);
                break;

            case MapNodeType.Shop:
                EnterShop(node);
                break;

            case MapNodeType.Event:
                EnterEvent(node);
                break;

            case MapNodeType.Treasure:
                OpenTreasure(node);
                break;
        }
    }

    #endregion

    #region Combat

    private void StartCombat(MapNode node)
    {
        GameLogger.Info("MainGameController", $"Starting combat: {node.NodeType}");
        _currentScreen = GameScreen.Combat;

        _mapContainer.Visible = false;
        _combatContainer.Visible = true;

        // Get AI depth from current map config
        int aiDepth = _runManager.GetCurrentAiDepth();

        // Adjust for node type (elites/bosses get +1 depth)
        if (node.NodeType == MapNodeType.Elite)
            aiDepth = Math.Min(aiDepth + 1, 5);
        else if (node.NodeType == MapNodeType.Boss)
            aiDepth = Math.Min(aiDepth + 1, 5);

        GameLogger.Info("MainGameController", $"Combat AI depth: {aiDepth}-ply (node type: {node.NodeType})");

        _combatController = new GameController();
        _combatController.SetAiDepth(aiDepth);
        _combatController.CombatMatchEnded += OnCombatMatchEndedSignal;
        _combatContainer.AddChild(_combatController);
    }

    private void OnCombatMatchEndedSignal(int winnerTeam)
    {
        var winner = (Team)winnerTeam;
        GameLogger.Info("MainGameController", $"Combat match ended signal received - Winner: {winner}");
        OnCombatMatchEnded(winner);
    }

    private void EndCombat(bool victory, int coinsEarned)
    {
        GameLogger.Info("MainGameController", $"Combat ended: Victory={victory}, Coins={coinsEarned}");

        if (_combatController != null)
        {
            _combatController.QueueFree();
            _combatController = null;
        }

        _runManager.CompleteCurrentNode(victory, coinsEarned);

        if (victory)
        {
            ShowMapScreen();
            AutoSave();
        }
    }

    private void EnterShop(MapNode node)
    {
        GameLogger.Info("MainGameController", "Entering shop (TODO: implement shop UI)");
        _currentScreen = GameScreen.Shop;
        _runManager.CompleteCurrentNode(true, 0);
        AutoSave();
    }

    private void EnterEvent(MapNode node)
    {
        GameLogger.Info("MainGameController", "Entering event (TODO: implement event UI)");
        _currentScreen = GameScreen.Event;
        int coins = GD.RandRange(5, 15);
        _runManager.CompleteCurrentNode(true, coins);
        AutoSave();
    }

    private void OpenTreasure(MapNode node)
    {
        GameLogger.Info("MainGameController", "Opening treasure (TODO: implement treasure UI)");
        _currentScreen = GameScreen.Treasure;
        int coins = GD.RandRange(20, 40);
        _runManager.CompleteCurrentNode(true, coins);
        AutoSave();
    }

    #endregion

    #region Screen Management

    private void ShowMapScreen()
    {
        _currentScreen = GameScreen.Map;
        _mapContainer.Visible = true;
        _combatContainer.Visible = false;

        if (_combatController != null)
        {
            _combatController.QueueFree();
            _combatController = null;
        }

        _mapUI.SetMap(_runManager.CurrentActMap);
        GameLogger.Debug("MainGameController", "Showing map screen");
    }

    #endregion

    #region Run Events

    private void OnNodeCompleted(MapNode node)
    {
        GameLogger.Info("MainGameController", $"Node completed: {node.NodeType}");
        if (_currentScreen != GameScreen.Map)
        {
            ShowMapScreen();
        }
    }

    private void OnActCompleted(int actNumber)
    {
        GameLogger.Info("MainGameController", $"Act {actNumber} completed!");
        ShowMapScreen();
        AutoSave();
    }

    private void OnRunCompleted(bool victory)
    {
        GameLogger.Info("MainGameController", $"Run completed: {(victory ? "VICTORY!" : "DEFEAT")}");

        var label = new Label
        {
            Text = victory ? "VICTORY!\n\nYou conquered all acts!" : "DEFEAT\n\nYour King has fallen.",
            Position = new Vector2(400, 200)
        };
        label.AddThemeFontSizeOverride("font_size", 32);
        label.AddThemeColorOverride("font_color", victory ? Colors.Gold : Colors.Red);
        _overlayUI.AddChild(label);

        EmitSignal(SignalName.GameEnded, victory);
    }

    #endregion

    /// <summary>
    /// Called from GameController when combat ends.
    /// </summary>
    public void OnCombatMatchEnded(Team winner)
    {
        bool victory = winner == Team.Player;
        int coins = victory ? GD.RandRange(10, 30) : 0;
        EndCombat(victory, coins);
    }
}
