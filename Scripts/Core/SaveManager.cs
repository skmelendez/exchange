using Godot;
using System.Text.Json;
using System.Text.Json.Serialization;
using GodotFileAccess = Godot.FileAccess;

namespace Exchange.Core;

/// <summary>
/// Handles saving and loading game state to/from JSON.
/// Save file location: user://save.json
/// </summary>
public static class SaveManager
{
    private const string SaveFileName = "user://save.json";
    private const int CurrentSaveVersion = 1;

    /// <summary>
    /// Complete save data structure for the game.
    /// </summary>
    public class SaveData
    {
        /// <summary>Save format version for migration support.</summary>
        public int Version { get; set; } = CurrentSaveVersion;

        /// <summary>When the save was created.</summary>
        public string Timestamp { get; set; } = "";

        // Run state
        public int RunSeed { get; set; }
        public int CurrentActNumber { get; set; }
        public int Coins { get; set; }
        public int CurrentHealth { get; set; }
        public int MaxHealth { get; set; }
        public List<string> Relics { get; set; } = [];
        public int TotalCombatsWon { get; set; }
        public int TotalCoinsEarned { get; set; }

        // Map state
        public int? CurrentNodeId { get; set; }
        public List<int> VisitedNodeIds { get; set; } = [];

        // Game screen state
        public string GameScreen { get; set; } = "Map"; // "Map" or "Combat"

        // Combat state (only populated if in combat)
        public CombatSaveData? CombatState { get; set; }
    }

    /// <summary>
    /// Combat-specific save data for mid-combat saves.
    /// </summary>
    public class CombatSaveData
    {
        public string CurrentPhase { get; set; } = "PlayerTurn";
        public int TurnNumber { get; set; }
        public List<PieceSaveData> Pieces { get; set; } = [];

        // Modifiers
        public int PlayerDiceModifier { get; set; }
        public int EnemyDiceModifier { get; set; }
        public bool PlayerKingThreatened { get; set; }
        public bool EnemyKingThreatened { get; set; }
        public bool PlayerRoyalDecreeActive { get; set; }
        public bool EnemyRoyalDecreeActive { get; set; }
        public bool PlayerRoyalDecreeUsed { get; set; }
        public bool EnemyRoyalDecreeUsed { get; set; }
    }

    /// <summary>
    /// Individual piece state for combat saves.
    /// </summary>
    public class PieceSaveData
    {
        public string PieceType { get; set; } = "";
        public string Team { get; set; } = "";
        public int X { get; set; }
        public int Y { get; set; }
        public int CurrentHp { get; set; }
        public int MaxHp { get; set; }
        public int AbilityCooldownCurrent { get; set; }
        public bool AbilityUsedThisMatch { get; set; }
        public bool HasActedThisTurn { get; set; }
        public bool EnteredThreatZoneThisTurn { get; set; }

        // Piece-specific state
        public bool HasMovedThisTurn { get; set; } // Knight
        public bool UsedAdvanceLastTurn { get; set; } // Pawn
        public bool InterposeActive { get; set; } // Rook
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    /// <summary>
    /// Check if a save file exists.
    /// </summary>
    public static bool SaveExists()
    {
        return GodotFileAccess.FileExists(SaveFileName);
    }

    /// <summary>
    /// Save the current game state.
    /// </summary>
    /// <param name="saveData">The data to save</param>
    /// <returns>True if save succeeded</returns>
    public static bool Save(SaveData saveData)
    {
        try
        {
            saveData.Version = CurrentSaveVersion;
            saveData.Timestamp = DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss");

            string json = JsonSerializer.Serialize(saveData, JsonOptions);

            using var file = GodotFileAccess.Open(SaveFileName, GodotFileAccess.ModeFlags.Write);
            if (file == null)
            {
                GameLogger.Error("SaveManager", $"Failed to open save file for writing: {GodotFileAccess.GetOpenError()}");
                return false;
            }

            file.StoreString(json);
            GameLogger.Info("SaveManager", $"Game saved successfully at {saveData.Timestamp}");
            return true;
        }
        catch (Exception ex)
        {
            GameLogger.Exception("SaveManager", ex);
            return false;
        }
    }

    /// <summary>
    /// Load game state from save file.
    /// </summary>
    /// <returns>SaveData if successful, null if failed or no save exists</returns>
    public static SaveData? Load()
    {
        try
        {
            if (!SaveExists())
            {
                GameLogger.Info("SaveManager", "No save file found");
                return null;
            }

            using var file = GodotFileAccess.Open(SaveFileName, GodotFileAccess.ModeFlags.Read);
            if (file == null)
            {
                GameLogger.Error("SaveManager", $"Failed to open save file for reading: {GodotFileAccess.GetOpenError()}");
                return null;
            }

            string json = file.GetAsText();
            var saveData = JsonSerializer.Deserialize<SaveData>(json, JsonOptions);

            if (saveData == null)
            {
                GameLogger.Error("SaveManager", "Failed to deserialize save data");
                return null;
            }

            // Version migration if needed
            if (saveData.Version < CurrentSaveVersion)
            {
                GameLogger.Info("SaveManager", $"Migrating save from v{saveData.Version} to v{CurrentSaveVersion}");
                saveData = MigrateSave(saveData);
            }

            GameLogger.Info("SaveManager", $"Game loaded from {saveData.Timestamp}");
            return saveData;
        }
        catch (Exception ex)
        {
            GameLogger.Exception("SaveManager", ex);
            return null;
        }
    }

    /// <summary>
    /// Delete the save file.
    /// </summary>
    public static bool DeleteSave()
    {
        try
        {
            if (!SaveExists())
            {
                return true;
            }

            var error = DirAccess.RemoveAbsolute(ProjectSettings.GlobalizePath(SaveFileName));
            if (error != Error.Ok)
            {
                GameLogger.Error("SaveManager", $"Failed to delete save: {error}");
                return false;
            }

            GameLogger.Info("SaveManager", "Save file deleted");
            return true;
        }
        catch (Exception ex)
        {
            GameLogger.Exception("SaveManager", ex);
            return false;
        }
    }

    /// <summary>
    /// Get save info without loading full state (for menu display).
    /// </summary>
    public static (bool exists, string? timestamp, int? actNumber) GetSaveInfo()
    {
        if (!SaveExists())
        {
            return (false, null, null);
        }

        try
        {
            var saveData = Load();
            if (saveData != null)
            {
                return (true, saveData.Timestamp, saveData.CurrentActNumber);
            }
        }
        catch
        {
            // Ignore errors, just return no info
        }

        return (true, null, null);
    }

    /// <summary>
    /// Migrate old save formats to current version.
    /// </summary>
    private static SaveData MigrateSave(SaveData oldSave)
    {
        // Currently no migrations needed (v1 is first version)
        // Future migrations would go here:
        // if (oldSave.Version < 2) { ... migrate v1 to v2 ... }

        oldSave.Version = CurrentSaveVersion;
        return oldSave;
    }
}
