namespace Exchange.Core;

/// <summary>
/// Static data definitions for piece types.
/// HP and Base Damage are LOCKED per the design doc.
/// </summary>
public static class PieceData
{
    public record PieceStats(int MaxHp, int BaseDamage, AbilityId Ability, int AbilityCooldown);

    public static readonly Dictionary<PieceType, PieceStats> Stats = new()
    {
        { PieceType.King,   new PieceStats(25, 1, AbilityId.RoyalDecree, -1) },  // -1 = once per match, 25 HP prevents quick snipes
        { PieceType.Queen,  new PieceStats(10, 3, AbilityId.Overextend, 3) },
        { PieceType.Rook,   new PieceStats(13, 2, AbilityId.Interpose, 3) },
        { PieceType.Bishop, new PieceStats(10, 2, AbilityId.Consecration, 3) },
        { PieceType.Knight, new PieceStats(11, 2, AbilityId.Skirmish, 3) },
        { PieceType.Pawn,   new PieceStats(7, 1, AbilityId.Advance, 1) }
    };

    /// <summary>
    /// Standard chess notation for pieces
    /// </summary>
    public static char GetNotation(PieceType type, Team team)
    {
        char c = type switch
        {
            PieceType.King => 'K',
            PieceType.Queen => 'Q',
            PieceType.Rook => 'R',
            PieceType.Bishop => 'B',
            PieceType.Knight => 'N',
            PieceType.Pawn => 'P',
            _ => '?'
        };
        return team == Team.Player ? c : char.ToLower(c);
    }
}
