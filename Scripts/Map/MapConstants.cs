namespace Exchange.Map;

/// <summary>
/// Constants for map generation and act configuration.
/// All values are compile-time constants - not modifiable outside dev builds.
/// </summary>
public static class MapConstants
{
    // ===========================================
    // GLOBAL MAP SETTINGS
    // ===========================================

    /// <summary>Max vertical divergence (rows) for any act</summary>
    public const int GlobalMaxRows = 3;

    /// <summary>Minimum paths from start</summary>
    public const int GlobalMinStartingPaths = 1;

    /// <summary>Maximum paths from start</summary>
    public const int GlobalMaxStartingPaths = 3;

    // ===========================================
    // ACT 1 CONFIGURATION
    // ===========================================

    public const int Act1_Columns = 8;
    public const int Act1_MaxRows = 3;
    public const int Act1_MinStartingPaths = 2;
    public const int Act1_MaxStartingPaths = 3;
    public const int Act1_EliteColumn = 4;
    public const int Act1_AiDepth = 2;

    // Encounter type weights (must sum to 1.0)
    public const float Act1_CombatWeight = 0.55f;
    public const float Act1_EventWeight = 0.20f;
    public const float Act1_TreasureWeight = 0.15f;
    public const float Act1_ShopWeight = 0.10f;

    // ===========================================
    // ACT 2 CONFIGURATION
    // ===========================================

    public const int Act2_Columns = 9;
    public const int Act2_MaxRows = 3;
    public const int Act2_MinStartingPaths = 2;
    public const int Act2_MaxStartingPaths = 3;
    public const int Act2_EliteColumn = 5;
    public const int Act2_AiDepth = 3;

    // Encounter type weights
    public const float Act2_CombatWeight = 0.60f;
    public const float Act2_EventWeight = 0.18f;
    public const float Act2_TreasureWeight = 0.12f;
    public const float Act2_ShopWeight = 0.10f;

    // ===========================================
    // ACT 3 CONFIGURATION
    // ===========================================

    public const int Act3_Columns = 10;
    public const int Act3_MaxRows = 3;
    public const int Act3_MinStartingPaths = 2;
    public const int Act3_MaxStartingPaths = 3;
    public const int Act3_EliteColumn = 5;
    public const int Act3_AiDepth = 4;

    // Encounter type weights
    public const float Act3_CombatWeight = 0.65f;
    public const float Act3_EventWeight = 0.15f;
    public const float Act3_TreasureWeight = 0.10f;
    public const float Act3_ShopWeight = 0.10f;

    // ===========================================
    // FINAL BOSS CONFIGURATION
    // ===========================================

    public const int Final_Columns = 2;  // Just start -> boss
    public const int Final_MaxRows = 1;
    public const int Final_MinStartingPaths = 1;
    public const int Final_MaxStartingPaths = 1;
    public const int Final_EliteColumn = -1;  // No elite
    public const int Final_AiDepth = 4;

    // ===========================================
    // REWARD CONFIGURATION
    // ===========================================

    // Combat rewards (coins)
    public const int CombatReward_Min = 10;
    public const int CombatReward_Max = 25;

    public const int EliteReward_Min = 25;
    public const int EliteReward_Max = 50;

    public const int BossReward_Min = 75;
    public const int BossReward_Max = 125;

    // Treasure rewards
    public const int TreasureCoins_Min = 20;
    public const int TreasureCoins_Max = 40;

    // Event rewards (can vary by event)
    public const int EventCoins_Min = 5;
    public const int EventCoins_Max = 20;

    // ===========================================
    // HELPER METHODS
    // ===========================================

    /// <summary>
    /// Get ActConfig for a specific act number
    /// </summary>
    public static ActConfig GetActConfig(int actNumber) => actNumber switch
    {
        1 => new ActConfig
        {
            ActNumber = 1,
            Columns = Act1_Columns,
            MaxRows = Act1_MaxRows,
            MinStartingPaths = Act1_MinStartingPaths,
            MaxStartingPaths = Act1_MaxStartingPaths,
            EliteColumn = Act1_EliteColumn,
            AiDepth = Act1_AiDepth,
            CombatWeight = Act1_CombatWeight,
            EventWeight = Act1_EventWeight,
            TreasureWeight = Act1_TreasureWeight,
            ShopWeight = Act1_ShopWeight
        },
        2 => new ActConfig
        {
            ActNumber = 2,
            Columns = Act2_Columns,
            MaxRows = Act2_MaxRows,
            MinStartingPaths = Act2_MinStartingPaths,
            MaxStartingPaths = Act2_MaxStartingPaths,
            EliteColumn = Act2_EliteColumn,
            AiDepth = Act2_AiDepth,
            CombatWeight = Act2_CombatWeight,
            EventWeight = Act2_EventWeight,
            TreasureWeight = Act2_TreasureWeight,
            ShopWeight = Act2_ShopWeight
        },
        3 => new ActConfig
        {
            ActNumber = 3,
            Columns = Act3_Columns,
            MaxRows = Act3_MaxRows,
            MinStartingPaths = Act3_MinStartingPaths,
            MaxStartingPaths = Act3_MaxStartingPaths,
            EliteColumn = Act3_EliteColumn,
            AiDepth = Act3_AiDepth,
            CombatWeight = Act3_CombatWeight,
            EventWeight = Act3_EventWeight,
            TreasureWeight = Act3_TreasureWeight,
            ShopWeight = Act3_ShopWeight
        },
        4 => new ActConfig
        {
            ActNumber = 4,
            Columns = Final_Columns,
            MaxRows = Final_MaxRows,
            MinStartingPaths = Final_MinStartingPaths,
            MaxStartingPaths = Final_MaxStartingPaths,
            EliteColumn = Final_EliteColumn,
            AiDepth = Final_AiDepth,
            CombatWeight = 1.0f,
            EventWeight = 0f,
            TreasureWeight = 0f,
            ShopWeight = 0f
        },
        _ => throw new ArgumentOutOfRangeException(nameof(actNumber), $"Invalid act number: {actNumber}")
    };

    /// <summary>
    /// Get all act configs
    /// </summary>
    public static ActConfig[] GetAllActConfigs() => new[]
    {
        GetActConfig(1),
        GetActConfig(2),
        GetActConfig(3),
        GetActConfig(4)
    };

    /// <summary>
    /// Get reward for completing a node type
    /// </summary>
    public static (int min, int max) GetRewardRange(MapNodeType nodeType) => nodeType switch
    {
        MapNodeType.Combat => (CombatReward_Min, CombatReward_Max),
        MapNodeType.Elite => (EliteReward_Min, EliteReward_Max),
        MapNodeType.Boss => (BossReward_Min, BossReward_Max),
        MapNodeType.Treasure => (TreasureCoins_Min, TreasureCoins_Max),
        MapNodeType.Event => (EventCoins_Min, EventCoins_Max),
        _ => (0, 0)
    };
}
