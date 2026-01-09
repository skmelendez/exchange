using Godot;
using Exchange.Core;

namespace Exchange.UI;

/// <summary>
/// Main menu displayed on game launch.
/// Options: New Game, Continue (if save exists), Quit.
/// </summary>
public partial class MainMenu : CanvasLayer
{
    [Signal] public delegate void NewGameRequestedEventHandler();
    [Signal] public delegate void ContinueRequestedEventHandler();
    [Signal] public delegate void QuitRequestedEventHandler();

    private Button _newGameButton = null!;
    private Button _continueButton = null!;
    private Button _quitButton = null!;
    private Label _saveInfoLabel = null!;
    private Label _titleLabel = null!;

    public override void _Ready()
    {
        Layer = 50; // Above game, below pause menu
        BuildUI();
        UpdateContinueButton();

        GameLogger.Info("MainMenu", "Main menu displayed");
    }

    private void BuildUI()
    {
        // Full-screen dark background
        var background = new ColorRect
        {
            Color = new Color(0.08f, 0.08f, 0.12f, 1f),
            AnchorsPreset = (int)Control.LayoutPreset.FullRect
        };
        AddChild(background);

        // Center container
        var centerContainer = new CenterContainer
        {
            AnchorsPreset = (int)Control.LayoutPreset.FullRect
        };
        AddChild(centerContainer);

        // Main panel
        var panel = new PanelContainer
        {
            CustomMinimumSize = new Vector2(400, 350)
        };
        centerContainer.AddChild(panel);

        // Content
        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 20);
        panel.AddChild(vbox);

        // Title
        _titleLabel = new Label
        {
            Text = "EXCHANGE",
            HorizontalAlignment = HorizontalAlignment.Center
        };
        _titleLabel.AddThemeFontSizeOverride("font_size", 48);
        _titleLabel.AddThemeColorOverride("font_color", new Color(0.9f, 0.8f, 0.3f));
        vbox.AddChild(_titleLabel);

        // Subtitle
        var subtitle = new Label
        {
            Text = "A Chess Roguelike",
            HorizontalAlignment = HorizontalAlignment.Center
        };
        subtitle.AddThemeFontSizeOverride("font_size", 16);
        subtitle.AddThemeColorOverride("font_color", Colors.LightGray);
        vbox.AddChild(subtitle);

        vbox.AddChild(new HSeparator());

        // New Game button
        _newGameButton = CreateMenuButton("New Game", new Color(0.3f, 0.7f, 0.3f));
        _newGameButton.Pressed += OnNewGamePressed;
        vbox.AddChild(_newGameButton);

        // Continue button
        _continueButton = CreateMenuButton("Continue", new Color(0.3f, 0.5f, 0.8f));
        _continueButton.Pressed += OnContinuePressed;
        vbox.AddChild(_continueButton);

        // Save info label (shows act number and timestamp)
        _saveInfoLabel = new Label
        {
            Text = "",
            HorizontalAlignment = HorizontalAlignment.Center
        };
        _saveInfoLabel.AddThemeFontSizeOverride("font_size", 12);
        _saveInfoLabel.AddThemeColorOverride("font_color", Colors.Gray);
        vbox.AddChild(_saveInfoLabel);

        vbox.AddChild(new HSeparator());

        // Quit button
        _quitButton = CreateMenuButton("Quit", new Color(0.7f, 0.3f, 0.3f));
        _quitButton.Pressed += OnQuitPressed;
        vbox.AddChild(_quitButton);

        // Version info at bottom
        var versionLabel = new Label
        {
            Text = "v0.1.0 - Development Build",
            HorizontalAlignment = HorizontalAlignment.Center
        };
        versionLabel.AddThemeFontSizeOverride("font_size", 10);
        versionLabel.AddThemeColorOverride("font_color", new Color(0.5f, 0.5f, 0.5f));
        vbox.AddChild(versionLabel);
    }

    private Button CreateMenuButton(string text, Color highlightColor)
    {
        var btn = new Button
        {
            Text = text,
            CustomMinimumSize = new Vector2(250, 50)
        };
        btn.AddThemeFontSizeOverride("font_size", 20);
        return btn;
    }

    /// <summary>
    /// Update the Continue button state based on save file existence.
    /// </summary>
    public void UpdateContinueButton()
    {
        var (exists, timestamp, actNumber) = SaveManager.GetSaveInfo();

        if (exists)
        {
            _continueButton.Disabled = false;
            _continueButton.Modulate = Colors.White;

            if (timestamp != null && actNumber != null)
            {
                string actText = actNumber == 4 ? "Final Boss" : $"Act {actNumber}";
                // Parse timestamp to show just date
                if (DateTime.TryParse(timestamp, out var dt))
                {
                    _saveInfoLabel.Text = $"{actText} - Saved {dt:MMM d, h:mm tt}";
                }
                else
                {
                    _saveInfoLabel.Text = $"{actText}";
                }
            }
            else
            {
                _saveInfoLabel.Text = "Save found";
            }
        }
        else
        {
            _continueButton.Disabled = true;
            _continueButton.Modulate = new Color(0.5f, 0.5f, 0.5f);
            _saveInfoLabel.Text = "No save found";
        }
    }

    private void OnNewGamePressed()
    {
        GameLogger.Info("MainMenu", "New Game selected");

        // If save exists, could show confirmation dialog
        // For now, just start new game (overwrites save on first node completion)
        EmitSignal(SignalName.NewGameRequested);
    }

    private void OnContinuePressed()
    {
        GameLogger.Info("MainMenu", "Continue selected");
        EmitSignal(SignalName.ContinueRequested);
    }

    private void OnQuitPressed()
    {
        GameLogger.Info("MainMenu", "Quit selected");
        EmitSignal(SignalName.QuitRequested);
    }

    /// <summary>
    /// Show the main menu.
    /// </summary>
    public new void Show()
    {
        Visible = true;
        UpdateContinueButton();
    }

    /// <summary>
    /// Hide the main menu.
    /// </summary>
    public new void Hide()
    {
        Visible = false;
    }
}
