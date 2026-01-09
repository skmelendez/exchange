using Godot;
using Exchange.Core;
using Exchange.Map;

namespace Exchange.UI;

/// <summary>
/// Debug menu for development testing.
/// Allows modifying map settings, regenerating maps, auto-winning, etc.
/// </summary>
public partial class DebugMenu : CanvasLayer
{
    [Signal] public delegate void CloseRequestedEventHandler();
    [Signal] public delegate void AutoWinRequestedEventHandler();
    [Signal] public delegate void RegenerateMapRequestedEventHandler(int seed);
    [Signal] public delegate void SkipToActRequestedEventHandler(int actNumber);

    private Control _panel = null!;
    private SpinBox _seedInput = null!;
    private SpinBox _columnsInput = null!;
    private SpinBox _rowsInput = null!;
    private SpinBox _aiDepthInput = null!;
    private Label _currentActLabel = null!;
    private Label _currentSeedLabel = null!;

    private RunManager? _runManager;
    private bool _isVisible = false;

    public new bool IsVisible => _isVisible;

    public void Initialize(RunManager runManager)
    {
        _runManager = runManager;
    }

    public override void _Ready()
    {
        Layer = 101;  // Above pause menu
        ProcessMode = ProcessModeEnum.Always;  // Work even when paused
        BuildUI();
        Hide();
    }

    private void BuildUI()
    {
        // Semi-transparent background
        var background = new ColorRect
        {
            Color = new Color(0, 0, 0, 0.8f),
            AnchorsPreset = (int)Control.LayoutPreset.FullRect
        };
        AddChild(background);

        // Main panel
        var scrollContainer = new ScrollContainer
        {
            AnchorsPreset = (int)Control.LayoutPreset.FullRect,
            CustomMinimumSize = new Vector2(500, 600)
        };

        var centerContainer = new CenterContainer
        {
            AnchorsPreset = (int)Control.LayoutPreset.FullRect
        };

        _panel = new PanelContainer
        {
            CustomMinimumSize = new Vector2(450, 550)
        };

        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 10);
        _panel.AddChild(vbox);
        centerContainer.AddChild(_panel);
        AddChild(centerContainer);

        // Title
        var title = new Label
        {
            Text = "DEBUG MENU",
            HorizontalAlignment = HorizontalAlignment.Center
        };
        title.AddThemeFontSizeOverride("font_size", 24);
        title.AddThemeColorOverride("font_color", new Color(1f, 0.8f, 0.2f));
        vbox.AddChild(title);

        vbox.AddChild(CreateSeparator());

        // Current state info
        _currentActLabel = new Label { Text = "Current Act: --" };
        _currentActLabel.AddThemeColorOverride("font_color", Colors.LightGray);
        vbox.AddChild(_currentActLabel);

        _currentSeedLabel = new Label { Text = "Current Seed: --" };
        _currentSeedLabel.AddThemeColorOverride("font_color", Colors.LightGray);
        vbox.AddChild(_currentSeedLabel);

        vbox.AddChild(CreateSeparator());

        // === COMBAT SECTION ===
        vbox.AddChild(CreateSectionLabel("COMBAT"));

        var autoWinBtn = CreateButton("Auto-Win Current Match", new Color(0.4f, 0.9f, 0.4f));
        autoWinBtn.Pressed += OnAutoWinPressed;
        vbox.AddChild(autoWinBtn);

        vbox.AddChild(CreateSeparator());

        // === MAP SETTINGS ===
        vbox.AddChild(CreateSectionLabel("MAP SETTINGS"));

        var seedRow = CreateInputRow("Seed (0 = random):", out _seedInput, 0, int.MaxValue, 0);
        _seedInput.CustomMinimumSize = new Vector2(180, 0); // Wider for large numbers
        vbox.AddChild(seedRow);

        var colRow = CreateInputRow("Columns:", out _columnsInput, 4, 15, 8);
        vbox.AddChild(colRow);

        var rowRow = CreateInputRow("Max Rows:", out _rowsInput, 1, 5, 3);
        vbox.AddChild(rowRow);

        var aiRow = CreateInputRow("AI Depth:", out _aiDepthInput, 1, 6, 2);
        vbox.AddChild(aiRow);

        var regenBtn = CreateButton("Regenerate Map", new Color(0.4f, 0.7f, 1f));
        regenBtn.Pressed += OnRegeneratePressed;
        vbox.AddChild(regenBtn);

        vbox.AddChild(CreateSeparator());

        // === ACT NAVIGATION ===
        vbox.AddChild(CreateSectionLabel("SKIP TO ACT"));

        var actBtnRow = new HBoxContainer();
        actBtnRow.AddThemeConstantOverride("separation", 10);

        for (int i = 1; i <= 4; i++)
        {
            int actNum = i;
            var actLabel = i == 4 ? "Final" : $"Act {i}";
            var actBtn = CreateSmallButton(actLabel);
            actBtn.Pressed += () => OnSkipToActPressed(actNum);
            actBtnRow.AddChild(actBtn);
        }
        vbox.AddChild(actBtnRow);

        vbox.AddChild(CreateSeparator());

        // Close button
        var closeBtn = CreateButton("Close", Colors.White);
        closeBtn.Pressed += OnClosePressed;
        vbox.AddChild(closeBtn);
    }

    private HSeparator CreateSeparator()
    {
        return new HSeparator();
    }

    private Label CreateSectionLabel(string text)
    {
        var label = new Label
        {
            Text = text,
            HorizontalAlignment = HorizontalAlignment.Center
        };
        label.AddThemeFontSizeOverride("font_size", 16);
        label.AddThemeColorOverride("font_color", new Color(0.8f, 0.8f, 0.8f));
        return label;
    }

    private Button CreateButton(string text, Color color)
    {
        var btn = new Button
        {
            Text = text,
            CustomMinimumSize = new Vector2(300, 35)
        };
        btn.AddThemeFontSizeOverride("font_size", 14);
        return btn;
    }

    private Button CreateSmallButton(string text)
    {
        var btn = new Button
        {
            Text = text,
            CustomMinimumSize = new Vector2(80, 30)
        };
        btn.AddThemeFontSizeOverride("font_size", 12);
        return btn;
    }

    private HBoxContainer CreateInputRow(string label, out SpinBox spinBox, double min, double max, double defaultVal)
    {
        var row = new HBoxContainer();

        var lbl = new Label
        {
            Text = label,
            CustomMinimumSize = new Vector2(150, 0)
        };
        lbl.AddThemeColorOverride("font_color", Colors.LightGray);
        row.AddChild(lbl);

        spinBox = new SpinBox
        {
            MinValue = min,
            MaxValue = max,
            Value = defaultVal,
            CustomMinimumSize = new Vector2(100, 0)
        };
        row.AddChild(spinBox);

        return row;
    }

    public new void Show()
    {
        _isVisible = true;
        Visible = true;
        UpdateCurrentStateDisplay();
    }

    public new void Hide()
    {
        _isVisible = false;
        Visible = false;
    }

    private void UpdateCurrentStateDisplay()
    {
        if (_runManager == null)
        {
            GameLogger.Warning("DebugMenu", "UpdateCurrentStateDisplay: _runManager is null");
            return;
        }

        var actText = _runManager.CurrentActNumber == 4 ? "Final Boss" : $"Act {_runManager.CurrentActNumber}";
        _currentActLabel.Text = $"Current Act: {actText}";

        // Update inputs to match current config
        if (_runManager.CurrentActMap != null)
        {
            var currentSeed = _runManager.CurrentActMap.Seed;
            var config = _runManager.CurrentActMap.Config;
            GameLogger.Debug("DebugMenu", $"Updating display - Seed: {currentSeed}, Columns: {config.Columns}, Rows: {config.MaxRows}, AI: {config.AiDepth}");

            _currentSeedLabel.Text = $"Current Seed: {currentSeed}";
            _seedInput.SetValueNoSignal(currentSeed);
            _columnsInput.SetValueNoSignal(config.Columns);
            _rowsInput.SetValueNoSignal(config.MaxRows);
            _aiDepthInput.SetValueNoSignal(config.AiDepth);
        }
        else
        {
            GameLogger.Warning("DebugMenu", "UpdateCurrentStateDisplay: CurrentActMap is null");
            _currentSeedLabel.Text = "Current Seed: --";
        }
    }

    private void OnAutoWinPressed()
    {
        GameLogger.Info("DebugMenu", "Auto-Win requested");
        EmitSignal(SignalName.AutoWinRequested);
    }

    private void OnRegeneratePressed()
    {
        int seed = (int)_seedInput.Value;
        if (seed == 0)
            seed = (int)(GD.Randi() % 999999);

        GameLogger.Info("DebugMenu", $"Regenerate map - Seed: {seed}, Columns: {(int)_columnsInput.Value}, Rows: {(int)_rowsInput.Value}, AI: {(int)_aiDepthInput.Value}");
        EmitSignal(SignalName.RegenerateMapRequested, seed);
        UpdateCurrentStateDisplay();
    }

    private void OnSkipToActPressed(int actNumber)
    {
        GameLogger.Info("DebugMenu", $"Skip to Act {actNumber}");
        EmitSignal(SignalName.SkipToActRequested, actNumber);
        Hide();
    }

    private void OnClosePressed()
    {
        Hide();
        EmitSignal(SignalName.CloseRequested);
    }

    public override void _Input(InputEvent @event)
    {
        if (@event is InputEventKey key && key.Pressed && key.Keycode == Key.Escape && _isVisible)
        {
            Hide();
            EmitSignal(SignalName.CloseRequested);
            GetViewport().SetInputAsHandled();
        }
    }

    /// <summary>
    /// Get the custom map settings from the UI
    /// </summary>
    public (int columns, int rows, int aiDepth) GetMapSettings()
    {
        return ((int)_columnsInput.Value, (int)_rowsInput.Value, (int)_aiDepthInput.Value);
    }
}
