using Godot;
using System.IO;

namespace Exchange.Core;

/// <summary>
/// Static file-based logger for debugging and tracking game state.
/// Writes to game_logs/ folder with timestamped files.
/// Also outputs to Godot console via GD.Print.
/// </summary>
public static class GameLogger
{
    /// <summary>Log severity levels.</summary>
    public enum Level
    {
        Debug,
        Info,
        Warning,
        Error
    }

    private static StreamWriter? _writer;
    private static string? _logFilePath;
    private static readonly object _lock = new();
    private static bool _initialized;

    /// <summary>Minimum level to log. Messages below this level are ignored.</summary>
    public static Level MinLevel { get; set; } = Level.Debug;

    /// <summary>Whether to also output to Godot console.</summary>
    public static bool OutputToConsole { get; set; } = true;

    /// <summary>
    /// Initialize the logger. Creates game_logs folder and opens log file.
    /// Call once at game startup.
    /// </summary>
    public static void Initialize()
    {
        lock (_lock)
        {
            if (_initialized) return;

            try
            {
                // Create game_logs folder in project directory
                string logsFolder = ProjectSettings.GlobalizePath("res://game_logs");

                if (!Directory.Exists(logsFolder))
                {
                    Directory.CreateDirectory(logsFolder);
                }

                // Create timestamped log file
                string timestamp = DateTime.Now.ToString("yyyy-MM-dd_HHmmss");
                _logFilePath = Path.Combine(logsFolder, $"game_{timestamp}.log");

                _writer = new StreamWriter(_logFilePath, append: false)
                {
                    AutoFlush = true // Flush on each write for crash safety
                };

                _initialized = true;

                // Log startup
                Info("GameLogger", $"Log file created: {_logFilePath}");
                Info("GameLogger", $"Exchange Game Started - {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                Info("GameLogger", new string('=', 60));
            }
            catch (Exception ex)
            {
                GD.PrintErr($"[GameLogger] Failed to initialize: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Log a message with the specified level and context.
    /// </summary>
    /// <param name="level">Severity level</param>
    /// <param name="context">Source context (e.g., "Combat", "AI", "Save")</param>
    /// <param name="message">Message to log</param>
    public static void Log(Level level, string context, string message)
    {
        if (level < MinLevel) return;

        lock (_lock)
        {
            string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            string levelStr = level switch
            {
                Level.Debug => "DBG",
                Level.Info => "INF",
                Level.Warning => "WRN",
                Level.Error => "ERR",
                _ => "???"
            };

            string formattedMessage = $"[{timestamp}] [{levelStr}] [{context}] {message}";

            // Write to file
            if (_initialized && _writer != null)
            {
                try
                {
                    _writer.WriteLine(formattedMessage);
                }
                catch (Exception ex)
                {
                    GD.PrintErr($"[GameLogger] Write failed: {ex.Message}");
                }
            }

            // Output to Godot console
            if (OutputToConsole)
            {
                switch (level)
                {
                    case Level.Error:
                        GD.PrintErr(formattedMessage);
                        break;
                    case Level.Warning:
                        GD.PushWarning(formattedMessage);
                        GD.Print(formattedMessage);
                        break;
                    default:
                        GD.Print(formattedMessage);
                        break;
                }
            }
        }
    }

    /// <summary>Log a debug message.</summary>
    public static void Debug(string context, string message) => Log(Level.Debug, context, message);

    /// <summary>Log an info message.</summary>
    public static void Info(string context, string message) => Log(Level.Info, context, message);

    /// <summary>Log a warning message.</summary>
    public static void Warning(string context, string message) => Log(Level.Warning, context, message);

    /// <summary>Log an error message.</summary>
    public static void Error(string context, string message) => Log(Level.Error, context, message);

    /// <summary>Log an exception with stack trace.</summary>
    public static void Exception(string context, Exception ex)
    {
        Error(context, $"Exception: {ex.Message}");
        Debug(context, $"Stack trace:\n{ex.StackTrace}");
    }

    /// <summary>
    /// Log a state change (useful for tracking game state).
    /// </summary>
    public static void StateChange(string context, string property, object? oldValue, object? newValue)
    {
        Debug(context, $"{property}: {oldValue ?? "null"} -> {newValue ?? "null"}");
    }

    /// <summary>
    /// Close the log file. Call on game exit.
    /// </summary>
    public static void Shutdown()
    {
        lock (_lock)
        {
            if (_writer != null)
            {
                Info("GameLogger", new string('=', 60));
                Info("GameLogger", $"Exchange Game Ended - {DateTime.Now:yyyy-MM-dd HH:mm:ss}");

                _writer.Flush();
                _writer.Close();
                _writer.Dispose();
                _writer = null;
            }
            _initialized = false;
        }
    }

    /// <summary>
    /// Get the current log file path (for debugging).
    /// </summary>
    public static string? GetLogFilePath() => _logFilePath;
}
