using Godot;
using Exchange.Core;
using Exchange.Map;
using Exchange.Pieces;

namespace Exchange.Combat;

/// <summary>
/// Defines act-specific enemy compositions, elite abilities, boss modifiers, and buff pools.
/// This system scales difficulty and variety across the game's acts.
/// </summary>
public static class EnemyConfiguration
{
    #region Act-Specific Enemy Compositions

    /// <summary>
    /// Get the enemy composition for a specific act and node type.
    /// </summary>
    public static EnemyComposition GetComposition(int actNumber, MapNodeType nodeType)
    {
        return nodeType switch
        {
            MapNodeType.Boss => GetBossComposition(actNumber),
            MapNodeType.Elite => GetEliteComposition(actNumber),
            _ => GetStandardComposition(actNumber)
        };
    }

    private static EnemyComposition GetStandardComposition(int actNumber)
    {
        return actNumber switch
        {
            1 => new EnemyComposition(
                // Act 1: Basic enemies, standard pieces
                AiDepth: 2,
                PieceTypes: new[] { PieceType.Pawn, PieceType.Pawn, PieceType.Knight, PieceType.Bishop },
                HpModifier: 1.0f,
                DamageModifier: 1.0f
            ),
            2 => new EnemyComposition(
                // Act 2: More pieces, better AI
                AiDepth: 3,
                PieceTypes: new[] { PieceType.Pawn, PieceType.Pawn, PieceType.Knight, PieceType.Knight, PieceType.Rook },
                HpModifier: 1.1f,
                DamageModifier: 1.0f
            ),
            3 => new EnemyComposition(
                // Act 3: Strong composition
                AiDepth: 4,
                PieceTypes: new[] { PieceType.Pawn, PieceType.Knight, PieceType.Bishop, PieceType.Rook, PieceType.Queen },
                HpModifier: 1.2f,
                DamageModifier: 1.1f
            ),
            _ => new EnemyComposition(
                // Act 4+: Full strength
                AiDepth: 5,
                PieceTypes: new[] { PieceType.Pawn, PieceType.Pawn, PieceType.Knight, PieceType.Bishop, PieceType.Rook, PieceType.Queen },
                HpModifier: 1.3f,
                DamageModifier: 1.2f
            )
        };
    }

    private static EnemyComposition GetEliteComposition(int actNumber)
    {
        return actNumber switch
        {
            1 => new EnemyComposition(
                AiDepth: 3,
                PieceTypes: new[] { PieceType.Pawn, PieceType.Knight, PieceType.Knight, PieceType.Bishop, PieceType.Rook },
                HpModifier: 1.2f,
                DamageModifier: 1.1f,
                EliteAbility: EliteAbility.Fortified // +2 HP to all pieces
            ),
            2 => new EnemyComposition(
                AiDepth: 4,
                PieceTypes: new[] { PieceType.Pawn, PieceType.Knight, PieceType.Bishop, PieceType.Rook, PieceType.Rook },
                HpModifier: 1.3f,
                DamageModifier: 1.15f,
                EliteAbility: EliteAbility.Aggressive // +1 damage to all attacks
            ),
            3 => new EnemyComposition(
                AiDepth: 5,
                PieceTypes: new[] { PieceType.Knight, PieceType.Knight, PieceType.Bishop, PieceType.Rook, PieceType.Queen },
                HpModifier: 1.4f,
                DamageModifier: 1.2f,
                EliteAbility: EliteAbility.Coordinated // Pieces near allies get +1 defense
            ),
            _ => new EnemyComposition(
                AiDepth: 6,
                PieceTypes: new[] { PieceType.Knight, PieceType.Bishop, PieceType.Rook, PieceType.Rook, PieceType.Queen },
                HpModifier: 1.5f,
                DamageModifier: 1.25f,
                EliteAbility: EliteAbility.Veteran // All elite abilities combined
            )
        };
    }

    private static EnemyComposition GetBossComposition(int actNumber)
    {
        return actNumber switch
        {
            1 => new EnemyComposition(
                AiDepth: 4,
                PieceTypes: new[] { PieceType.Pawn, PieceType.Pawn, PieceType.Knight, PieceType.Bishop, PieceType.Rook, PieceType.Queen },
                HpModifier: 1.5f,
                DamageModifier: 1.2f,
                BossModifier: BossModifier.ThroneGuard // King cannot be attacked unless alone
            ),
            2 => new EnemyComposition(
                AiDepth: 5,
                PieceTypes: new[] { PieceType.Pawn, PieceType.Knight, PieceType.Knight, PieceType.Rook, PieceType.Rook, PieceType.Queen },
                HpModifier: 1.7f,
                DamageModifier: 1.3f,
                BossModifier: BossModifier.Reinforcements // Spawn a pawn every 5 turns
            ),
            3 => new EnemyComposition(
                AiDepth: 5,
                PieceTypes: new[] { PieceType.Knight, PieceType.Bishop, PieceType.Bishop, PieceType.Rook, PieceType.Queen, PieceType.Queen },
                HpModifier: 1.8f,
                DamageModifier: 1.35f,
                BossModifier: BossModifier.Wrath // King deals double damage when below 50% HP
            ),
            _ => new EnemyComposition(
                // Final Boss
                AiDepth: 6,
                PieceTypes: new[] { PieceType.Knight, PieceType.Knight, PieceType.Rook, PieceType.Rook, PieceType.Queen, PieceType.Queen },
                HpModifier: 2.0f,
                DamageModifier: 1.5f,
                BossModifier: BossModifier.Ascended // All boss modifiers + regeneration
            )
        };
    }

    #endregion

    #region Buff Pools

    /// <summary>
    /// Get random buffs for enemies based on act.
    /// </summary>
    public static List<EnemyBuff> GetRandomBuffs(int actNumber, int count = 1)
    {
        var availableBuffs = GetBuffPoolForAct(actNumber);
        var selectedBuffs = new List<EnemyBuff>();
        var rng = new RandomNumberGenerator();

        for (int i = 0; i < count && availableBuffs.Count > 0; i++)
        {
            int index = rng.RandiRange(0, availableBuffs.Count - 1);
            selectedBuffs.Add(availableBuffs[index]);
            availableBuffs.RemoveAt(index); // No duplicates
        }

        return selectedBuffs;
    }

    private static List<EnemyBuff> GetBuffPoolForAct(int actNumber)
    {
        var buffs = new List<EnemyBuff>();

        // Base buffs available in all acts
        buffs.Add(new EnemyBuff("Tough", BuffType.HpBonus, 2));
        buffs.Add(new EnemyBuff("Sharp", BuffType.DamageBonus, 1));

        if (actNumber >= 2)
        {
            buffs.Add(new EnemyBuff("Armored", BuffType.HpBonus, 3));
            buffs.Add(new EnemyBuff("Precise", BuffType.DamageBonus, 2));
            buffs.Add(new EnemyBuff("Swift", BuffType.Initiative, 1)); // Acts first
        }

        if (actNumber >= 3)
        {
            buffs.Add(new EnemyBuff("Ironclad", BuffType.HpBonus, 5));
            buffs.Add(new EnemyBuff("Deadly", BuffType.DamageBonus, 3));
            buffs.Add(new EnemyBuff("Regenerating", BuffType.Regeneration, 1)); // Heal 1 HP per turn
            buffs.Add(new EnemyBuff("Vengeful", BuffType.Retaliation, 2)); // Deal 2 damage when hit
        }

        if (actNumber >= 4)
        {
            buffs.Add(new EnemyBuff("Titanic", BuffType.HpBonus, 8));
            buffs.Add(new EnemyBuff("Executioner", BuffType.DamageBonus, 4));
            buffs.Add(new EnemyBuff("Undying", BuffType.Regeneration, 2));
            buffs.Add(new EnemyBuff("Thorns", BuffType.Retaliation, 3));
            buffs.Add(new EnemyBuff("Commander", BuffType.AuraBonus, 1)); // Nearby allies +1 damage
        }

        return buffs;
    }

    #endregion
}

#region Data Types

public record EnemyComposition(
    int AiDepth,
    PieceType[] PieceTypes,
    float HpModifier,
    float DamageModifier,
    EliteAbility? EliteAbility = null,
    BossModifier? BossModifier = null
);

public record EnemyBuff(
    string Name,
    BuffType Type,
    int Value
);

public enum EliteAbility
{
    None,
    Fortified,      // +2 HP to all pieces
    Aggressive,     // +1 damage to all attacks
    Coordinated,    // Pieces near allies get +1 defense
    Veteran         // All abilities combined
}

public enum BossModifier
{
    None,
    ThroneGuard,    // King cannot be attacked unless alone
    Reinforcements, // Spawn a pawn every 5 turns
    Wrath,          // King deals double damage when below 50% HP
    Ascended        // All modifiers + regeneration
}

public enum BuffType
{
    HpBonus,
    DamageBonus,
    Initiative,
    Regeneration,
    Retaliation,
    AuraBonus
}

#endregion
