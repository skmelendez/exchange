namespace Exchange.Core;

public enum PieceType
{
    King,
    Queen,
    Rook,
    Bishop,
    Knight,
    Pawn
}

public enum Team
{
    Player,
    Enemy
}

public enum GamePhase
{
    PlayerTurn,
    EnemyTurn,
    Combat,
    AbilityResolution,
    CheckWinLoss,
    MatchEnd,
    Shop,
    RunComplete
}

public enum ActionType
{
    None,
    Move,
    Attack,
    Ability
}

public enum AbilityId
{
    // King
    RoyalDecree,        // Once per match: all allied combat rolls +1 until next turn

    // Queen
    Overextend,         // 3 turn CD: Move then attack, take 2 damage after

    // Rook
    Interpose,          // 3 turn CD: Damage to adjacent allies split with Rook

    // Bishop
    Consecration,       // 3 turn CD: Heal diagonal ally 1d6 HP

    // Knight
    Skirmish,           // 3 turn CD: Attack then reposition 1 tile

    // Pawn
    Advance             // 1 turn CD: Move forward one extra tile, no consecutive use
}

public enum MatchType
{
    Entry,
    Mid,
    Boss
}

/// <summary>
/// Tracks multi-step ability execution phases
/// </summary>
public enum AbilityPhase
{
    None,

    // Overextend (Queen): Move -> Attack -> Self-damage
    OverextendSelectMove,
    OverextendSelectAttack,

    // Skirmish (Knight): Attack -> Reposition
    SkirmishSelectAttack,
    SkirmishSelectReposition
}
