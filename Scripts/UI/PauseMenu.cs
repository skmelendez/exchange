using Godot;
using Exchange.Core;

namespace Exchange.UI;

/// <summary>
/// Pause menu shown when pressing ESC.
/// Options: Resume, Debug Menu, Save & Quit to Menu
/// </summary>
public partial class PauseMenu : CanvasLayer
{
    [Signal] public delegate void ResumeRequestedEventHandler();
    [Signal] public delegate void DebugMenuRequestedEventHandler();
    [Signal] public delegate void SaveAndQuitRequestedEventHandler();

    private Control _panel = null!;
    private bool _isVisible;

    public new bool IsVisible => _isVisible;

    public override void _Ready()
    {
        Layer = 100;  // Always on top
        BuildUI();
        Hide();
    }

    private void BuildUI()
    {
        // Semi-transparent background
        var background = new ColorRect
        {
            Color = new Color(0, 0, 0, 0.7f),
            AnchorsPreset = (int)Control.LayoutPreset.FullRect
        };
        AddChild(background);

        // Center panel
        _panel = new PanelContainer
        {
            CustomMinimumSize = new Vector2(300, 250)
        };

        // Center the panel
        var centerContainer = new CenterContainer
        {
            AnchorsPreset = (int)Control.LayoutPreset.FullRect
        };
        centerContainer.AddChild(_panel);
        AddChild(centerContainer);

        // Content
        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 15);
        _panel.AddChild(vbox);

        // Title
        var title = new Label
        {
            Text = "PAUSED",
            HorizontalAlignment = HorizontalAlignment.Center
        };
        title.AddThemeFontSizeOverride("font_size", 28);
        title.AddThemeColorOverride("font_color", Colors.White);
        vbox.AddChild(title);

        vbox.AddChild(new HSeparator());

        // Resume button
        var resumeBtn = CreateButton("Resume", Colors.White);
        resumeBtn.Pressed += OnResumePressed;
        vbox.AddChild(resumeBtn);

        // Debug Menu button
        var debugBtn = CreateButton("Debug Menu", new Color(1f, 0.8f, 0.2f));
        debugBtn.Pressed += OnDebugMenuPressed;
        vbox.AddChild(debugBtn);

        vbox.AddChild(new HSeparator());

        // Save & Quit button
        var saveQuitBtn = CreateButton("Save & Quit to Menu", new Color(0.9f, 0.6f, 0.2f));
        saveQuitBtn.Pressed += OnSaveAndQuitPressed;
        vbox.AddChild(saveQuitBtn);
    }

    private Button CreateButton(string text, Color color)
    {
        var btn = new Button
        {
            Text = text,
            CustomMinimumSize = new Vector2(200, 40)
        };
        btn.AddThemeFontSizeOverride("font_size", 18);
        return btn;
    }

    public new void Show()
    {
        _isVisible = true;
        Visible = true;
        GetTree().Paused = true;
    }

    public new void Hide()
    {
        _isVisible = false;
        Visible = false;
        GetTree().Paused = false;
    }

    public void Toggle()
    {
        if (_isVisible)
            Hide();
        else
            Show();
    }

    private void OnResumePressed()
    {
        Hide();
        EmitSignal(SignalName.ResumeRequested);
    }

    private void OnDebugMenuPressed()
    {
        GameLogger.Debug("PauseMenu", "Debug menu requested");
        EmitSignal(SignalName.DebugMenuRequested);
    }

    private void OnSaveAndQuitPressed()
    {
        GameLogger.Info("PauseMenu", "Save & Quit requested");
        Hide();
        EmitSignal(SignalName.SaveAndQuitRequested);
    }

    // Note: ESC handling removed - GameManager handles ESC key hierarchy
}
