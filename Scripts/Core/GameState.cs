using Godot;

namespace Exchange.Core;

/// <summary>
/// Central game state container. Tracks run progression, match state, and modifiers.
/// </summary>
public partial class GameState : Node
{
    // Run progression
    public int CurrentRoom { get; set; } = 1;
    public int CurrentMatch { get; set; } = 1;  // 1=Entry, 2=Mid, 3=Boss
    public int TotalMatchesWon { get; set; } = 0;
    public int Coins { get; set; } = 0;

    // Match state
    public GamePhase CurrentPhase { get; set; } = GamePhase.PlayerTurn;
    public Team CurrentTeam => CurrentPhase == GamePhase.PlayerTurn ? Team.Player : Team.Enemy;
    public int TurnNumber { get; set; } = 1;

    // Draw condition tracking
    public const int DrawMovesWithoutDamage = 30;  // Draw after 30 moves with no damage
    public int MovesWithoutDamage { get; set; } = 0;

    // Combat modifiers (reset each match)
    public int PlayerDiceModifier { get; set; } = 0;
    public int EnemyDiceModifier { get; set; } = 0;

    // King threat tracking (for +1 dice bonus)
    public bool PlayerKingThreatened { get; set; } = false;
    public bool EnemyKingThreatened { get; set; } = false;

    // Royal Decree tracking
    public bool PlayerRoyalDecreeActive { get; set; } = false;
    public bool EnemyRoyalDecreeActive { get; set; } = false;
    public bool PlayerRoyalDecreeUsed { get; set; } = false;
    public bool EnemyRoyalDecreeUsed { get; set; } = false;

    public MatchType GetMatchType() => CurrentMatch switch
    {
        1 => MatchType.Entry,
        2 => MatchType.Mid,
        3 => MatchType.Boss,
        _ => MatchType.Entry
    };

    public bool IsBossMatch => CurrentMatch == 3;

    /// <summary>
    /// Gets the active boss rule modifier for current room (only applies during boss matches)
    /// </summary>
    public int? GetActiveBossRule() => IsBossMatch ? CurrentRoom : null;

    /// <summary>
    /// Resets match-specific state for a new match
    /// </summary>
    public void ResetForNewMatch()
    {
        CurrentPhase = GamePhase.PlayerTurn;
        TurnNumber = 1;
        MovesWithoutDamage = 0;
        PlayerDiceModifier = 0;
        EnemyDiceModifier = 0;
        PlayerKingThreatened = false;
        EnemyKingThreatened = false;
        PlayerRoyalDecreeActive = false;
        EnemyRoyalDecreeActive = false;
        PlayerRoyalDecreeUsed = false;
        EnemyRoyalDecreeUsed = false;
    }

    /// <summary>
    /// Advances to next match/room
    /// </summary>
    public void AdvanceMatch()
    {
        TotalMatchesWon++;
        CurrentMatch++;
        if (CurrentMatch > 3)
        {
            CurrentMatch = 1;
            CurrentRoom++;
        }
    }

    public bool IsRunComplete => CurrentRoom > 5;
}
