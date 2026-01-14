namespace Exchange.Core;

/// <summary>
/// Static data definitions for piece types.
/// HP and Base Damage are LOCKED per the design doc.
/// </summary>
public static class PieceData
{
    /// <summary>
    /// Stats for a piece type.
    /// AbilityMaxUses: 0 = unlimited, >0 = limited uses per match
    /// </summary>
    public record PieceStats(int MaxHp, int BaseDamage, AbilityId Ability, int AbilityCooldown, int AbilityMaxUses);

    public static readonly Dictionary<PieceType, PieceStats> Stats = new()
    {
        { PieceType.King,   new PieceStats(25, 1, AbilityId.RoyalDecree, 0, 3) },   // 3 charges, no cooldown (+2 for 2 turns)
        { PieceType.Queen,  new PieceStats(10, 3, AbilityId.Overextend, 3, 0) },    // 0 = unlimited (self-limits via damage)
        { PieceType.Rook,   new PieceStats(13, 2, AbilityId.Interpose, 3, 5) },     // 5 uses (Black gets +1 in BasePiece)
        { PieceType.Bishop, new PieceStats(10, 2, AbilityId.Consecration, 3, 3) },  // 3 uses per match
        { PieceType.Knight, new PieceStats(11, 2, AbilityId.Skirmish, 3, 5) },      // 5 uses per match
        { PieceType.Pawn,   new PieceStats(7, 1, AbilityId.Advance, 1, 0) }         // 0 = unlimited
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
