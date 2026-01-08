using Godot;

namespace Exchange.Combat;

/// <summary>
/// Handles all dice rolling with full transparency as per design doc.
/// All rolls are 1d6 with various modifiers.
/// </summary>
public static class DiceRoller
{
    private static Random _random = new();

    public record DiceResult(int RawRoll, int Modifiers, int FinalValue, string Breakdown);

    /// <summary>
    /// Rolls 1d6 for combat with all applicable modifiers.
    /// </summary>
    public static DiceResult RollCombat(
        int baseModifier = 0,
        bool enteredThreatZone = false,
        bool royalDecreeActive = false,
        bool enemyKingWasThreatened = false)
    {
        int raw = Roll();
        int mods = baseModifier;
        var breakdown = new List<string> { $"Roll: {raw}" };

        // Threat zone penalty: -1 to piece's next combat roll
        if (enteredThreatZone)
        {
            mods -= 1;
            breakdown.Add("Threat Zone: -1");
        }

        // Royal Decree: +1 to all allied combat rolls
        if (royalDecreeActive)
        {
            mods += 1;
            breakdown.Add("Royal Decree: +1");
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
    /// Raw 1d6 roll
    /// </summary>
    public static int Roll() => _random.Next(1, 7);

    /// <summary>
    /// For testing/debugging: set a specific seed
    /// </summary>
    public static void SetSeed(int seed)
    {
        _random = new Random(seed);
    }
}
