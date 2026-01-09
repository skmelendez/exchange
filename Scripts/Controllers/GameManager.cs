using Godot;
using Exchange.Core;
using Exchange.UI;
using Exchange.Map;

namespace Exchange.Controllers;

/// <summary>
/// Root game controller that manages:
/// - Main menu display and flow
/// - Game session lifecycle (start, save, load, quit)
/// - ESC key handling hierarchy
/// - Transitions between menu and gameplay
/// </summary>
public partial class GameManager : Node2D
{
    private MainMenu _mainMenu = null!;
    private MainGameController? _gameController;
    private ColorRect _background = null!;

    private enum GameState { MainMenu, Playing }
    private GameState _state = GameState.MainMenu;

    public override void _Ready()
    {
        // Process input even when game tree is paused (for ESC handling)
        ProcessMode = ProcessModeEnum.Always;

        // Initialize logger first
        GameLogger.Initialize();
        GameLogger.Info("GameManager", "Game starting...");

        SetupBackground();
        SetupMainMenu();

        // Start at main menu
        ShowMainMenu();
    }

    public override void _ExitTree()
    {
        // Cleanup on exit
        GameLogger.Shutdown();
    }

    private void SetupBackground()
    {
        // Dark background that's always visible
        _background = new ColorRect
        {
            Name = "Background",
            Color = new Color(0.1f, 0.1f, 0.15f, 1f),
            AnchorsPreset = (int)Control.LayoutPreset.FullRect,
            ZIndex = -100
        };

        // Need a CanvasLayer to properly show the background
        var bgLayer = new CanvasLayer
        {
            Name = "BackgroundLayer",
            Layer = -10
        };
        AddChild(bgLayer);
        bgLayer.AddChild(_background);
    }

    private void SetupMainMenu()
    {
        _mainMenu = new MainMenu { Name = "MainMenu" };
        _mainMenu.NewGameRequested += OnNewGameRequested;
        _mainMenu.ContinueRequested += OnContinueRequested;
        _mainMenu.QuitRequested += OnQuitRequested;
        AddChild(_mainMenu);
    }

    public override void _Input(InputEvent @event)
    {
        // ESC handling at root level
        if (@event is InputEventKey key && key.Pressed && key.Keycode == Key.Escape)
        {
            HandleEscapeKey();
            GetViewport().SetInputAsHandled();
        }
    }

    private void HandleEscapeKey()
    {
        switch (_state)
        {
            case GameState.MainMenu:
                // ESC does nothing on main menu
                GameLogger.Debug("GameManager", "ESC pressed on main menu - ignored");
                break;

            case GameState.Playing:
                // Delegate to game controller
                if (_gameController != null)
                {
                    _gameController.HandleEscapeKey();
                }
                break;
        }
    }

    private void ShowMainMenu()
    {
        _state = GameState.MainMenu;
        _mainMenu.Show();
        GameLogger.Info("GameManager", "Showing main menu");
    }

    private void HideMainMenu()
    {
        _mainMenu.Hide();
    }

    private void OnNewGameRequested()
    {
        GameLogger.Info("GameManager", "Starting new game...");

        // Delete existing save (new game overwrites)
        if (SaveManager.SaveExists())
        {
            GameLogger.Info("GameManager", "Deleting existing save for new game");
            SaveManager.DeleteSave();
        }

        HideMainMenu();
        StartGame(null); // null = new game
    }

    private void OnContinueRequested()
    {
        GameLogger.Info("GameManager", "Loading saved game...");

        var saveData = SaveManager.Load();
        if (saveData == null)
        {
            GameLogger.Error("GameManager", "Failed to load save data!");
            // Stay on menu, refresh continue button state
            _mainMenu.UpdateContinueButton();
            return;
        }

        HideMainMenu();
        StartGame(saveData);
    }

    private void OnQuitRequested()
    {
        GameLogger.Info("GameManager", "Quit requested from main menu");
        GetTree().Quit();
    }

    private void StartGame(SaveManager.SaveData? saveData)
    {
        _state = GameState.Playing;

        // Create the main game controller
        _gameController = new MainGameController { Name = "MainGameController" };
        _gameController.SaveAndQuitRequested += OnSaveAndQuitRequested;
        _gameController.GameEnded += OnGameEnded;
        AddChild(_gameController);

        // Initialize with save data or start fresh
        if (saveData != null)
        {
            GameLogger.Info("GameManager", $"Initializing from save - Act {saveData.CurrentActNumber}");
            _gameController.InitializeFromSave(saveData);
        }
        else
        {
            GameLogger.Info("GameManager", "Initializing new run");
            _gameController.InitializeNewGame();
        }
    }

    private void OnSaveAndQuitRequested()
    {
        GameLogger.Info("GameManager", "Save & Quit requested");

        // Get save data from game controller
        if (_gameController != null)
        {
            var saveData = _gameController.CreateSaveData();
            if (saveData != null)
            {
                SaveManager.Save(saveData);
            }

            // Cleanup game controller
            _gameController.QueueFree();
            _gameController = null;
        }

        ShowMainMenu();
    }

    private void OnGameEnded(bool victory)
    {
        GameLogger.Info("GameManager", $"Game ended - Victory: {victory}");

        // On game end (victory or defeat), delete save and return to menu
        SaveManager.DeleteSave();

        if (_gameController != null)
        {
            _gameController.QueueFree();
            _gameController = null;
        }

        // Could show a results screen here
        // For now, go back to menu after a delay
        var timer = GetTree().CreateTimer(3.0f);
        timer.Timeout += ShowMainMenu;
    }

    /// <summary>
    /// Force save and quit (called from pause menu or window close).
    /// </summary>
    public void ForceSaveAndQuit()
    {
        if (_gameController != null && _state == GameState.Playing)
        {
            var saveData = _gameController.CreateSaveData();
            if (saveData != null)
            {
                SaveManager.Save(saveData);
            }
        }

        GameLogger.Info("GameManager", "Force quit - save completed");
        GetTree().Quit();
    }

    public override void _Notification(int what)
    {
        // Handle window close
        if (what == NotificationWMCloseRequest)
        {
            GameLogger.Info("GameManager", "Window close requested");
            ForceSaveAndQuit();
        }
    }
}
