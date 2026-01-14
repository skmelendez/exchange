using Godot;

namespace Exchange.Combat;

/// <summary>
/// Handles all dice rolling with full transparency as per design doc.
/// All rolls are 1d6 with various modifiers.
/// Uses Godot's built-in RNG for consistency with engine systems.
/// </summary>
public static class DiceRoller
{
    private static RandomNumberGenerator _rng = new();

    /// <summary>
    /// Result of a dice roll including breakdown for transparency.
    /// </summary>
    /// <param name="RawRoll">The unmodified die face (1-6)</param>
    /// <param name="Modifiers">Sum of all applied modifiers</param>
    /// <param name="FinalValue">Clamped result after modifiers</param>
    /// <param name="Breakdown">Human-readable breakdown string</param>
    public record DiceResult(int RawRoll, int Modifiers, int FinalValue, string Breakdown);

    /// <summary>
    /// Rolls 1d6 for combat with all applicable modifiers.
    /// </summary>
    /// <param name="threatPenalty">Threat zone penalty amount (normally -1, but -2 for Boss Room 1)</param>
    /// <param name="royalDecreeBonus">Royal Decree bonus (+2 when active, 0 otherwise)</param>
    public static DiceResult RollCombat(
        int baseModifier = 0,
        bool enteredThreatZone = false,
        int royalDecreeBonus = 0,
        bool enemyKingWasThreatened = false,
        int threatPenalty = -1)
    {
        int raw = Roll();
        int mods = baseModifier;
        var breakdown = new List<string> { $"Roll: {raw}" };

        // Threat zone penalty: normally -1, but -2 for Boss Room 1
        if (enteredThreatZone)
        {
            mods += threatPenalty;
            breakdown.Add($"Threat Zone: {threatPenalty}");
        }

        // Royal Decree: +2 to all allied combat rolls (3 charges, 2 turns duration)
        if (royalDecreeBonus > 0)
        {
            mods += royalDecreeBonus;
            breakdown.Add($"Royal Decree: +{royalDecreeBonus}");
        }

        // If enemy king was threatened at end of their turn, we get +1
        if (enemyKingWasThreatened)
        {
            mods += 1;
            breakdown.Add("King Threatened: +1");
        }

        if (baseModifier != 0)
        {
            breakdown.Add($"Base Modifier: {(baseModifier > 0 ? "+" : "")}{baseModifier}");
        }

        // Final value clamped to 1-6 range as per doc
        int final = Math.Clamp(raw + mods, 1, 6);
        breakdown.Add($"Final: {final}");

        return new DiceResult(raw, mods, final, string.Join(" | ", breakdown));
    }

    /// <summary>
    /// Rolls 1d6 for healing (Bishop's Consecration)
    /// </summary>
    public static DiceResult RollHealing(int baseModifier = 0)
    {
        int raw = Roll();
        int final = Math.Max(1, raw + baseModifier); // Healing can exceed 6
        var breakdown = $"Heal Roll: {raw}";
        if (baseModifier != 0)
            breakdown += $" + {baseModifier} = {final}";

        return new DiceResult(raw, baseModifier, final, breakdown);
    }

    /// <summary>
    /// Raw 1d6 roll using Godot's RandomNumberGenerator.
    /// </summary>
    /// <returns>Integer from 1 to 6 inclusive</returns>
    public static int Roll() => _rng.RandiRange(1, 6);

    /// <summary>
    /// Sets a specific seed for deterministic testing/debugging.
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public static void SetSeed(ulong seed)
    {
        _rng.Seed = seed;
    }

    /// <summary>
    /// Randomizes the seed using system time (default behavior).
    /// </summary>
    public static void Randomize()
    {
        _rng.Randomize();
    }
}
