using Godot;
using Exchange.Core;
using Exchange.Pieces;
using Exchange.Board;

namespace Exchange.Combat;

/// <summary>
/// Handles all combat resolution including attacks, damage calculation, and deaths.
/// </summary>
public partial class CombatResolver : Node
{
    // Using C# events for complex types (Godot signals don't support them)
    public event Action<CombatResult>? CombatResolved;
    public event Action<BasePiece>? PieceDestroyed;

    public record CombatResult(
        BasePiece Attacker,
        BasePiece Defender,
        int DamageDealt,
        bool DefenderDestroyed,
        DiceRoller.DiceResult DiceResult
    );

    private GameState? _gameState;

    public void Initialize(GameState gameState)
    {
        _gameState = gameState;
    }

    /// <summary>
    /// Resolves a base attack from attacker to defender
    /// </summary>
    public CombatResult ResolveAttack(BasePiece attacker, BasePiece defender, GameBoard board)
    {
        if (_gameState == null)
            throw new InvalidOperationException("CombatResolver not initialized");

        // Determine modifiers
        bool royalDecreeActive = attacker.Team == Team.Player
            ? _gameState.PlayerRoyalDecreeActive
            : _gameState.EnemyRoyalDecreeActive;

        bool enemyKingThreatened = attacker.Team == Team.Player
            ? _gameState.EnemyKingThreatened
            : _gameState.PlayerKingThreatened;

        // Roll dice with all modifiers
        var diceResult = DiceRoller.RollCombat(
            baseModifier: 0,
            enteredThreatZone: attacker.EnteredThreatZoneThisTurn,
            royalDecreeActive: royalDecreeActive,
            enemyKingWasThreatened: enemyKingThreatened
        );

        // Calculate total damage: Base Damage + Dice Roll
        int totalDamage = attacker.BaseDamage + diceResult.FinalValue;

        // Apply boss rules
        totalDamage = ApplyBossRules(totalDamage, attacker, defender, board);

        // Apply damage
        defender.TakeDamage(totalDamage);

        // Log combat
        GD.Print($"[Combat] {attacker.PieceType} attacks {defender.PieceType}: " +
                 $"Base {attacker.BaseDamage} + {diceResult.Breakdown} = {totalDamage} damage. " +
                 $"Defender HP: {defender.CurrentHp}/{defender.MaxHp}");

        bool destroyed = !defender.IsAlive;
        if (destroyed)
        {
            PieceDestroyed?.Invoke(defender);
            GD.Print($"[Combat] {defender.PieceType} destroyed!");
        }

        var result = new CombatResult(attacker, defender, totalDamage, destroyed, diceResult);
        CombatResolved?.Invoke(result);

        return result;
    }

    private int ApplyBossRules(int damage, BasePiece attacker, BasePiece defender, GameBoard board)
    {
        if (_gameState == null || !_gameState.IsBossMatch) return damage;

        int bossRule = _gameState.CurrentRoom;

        switch (bossRule)
        {
            case 2:
                // Room 2 Boss: Enemies adjacent to allies take -1 damage
                if (defender.Team == Team.Enemy && HasAdjacentAlly(defender, board))
                    damage = Math.Max(1, damage - 1);
                break;

            // Other boss rules don't directly affect damage calculation
        }

        return damage;
    }

    private bool HasAdjacentAlly(BasePiece piece, GameBoard board)
    {
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = piece.BoardPosition + dir;
            var adjacent = board.GetPieceAt(pos);
            if (adjacent != null && adjacent.Team == piece.Team && adjacent != piece)
                return true;
        }
        return false;
    }

    /// <summary>
    /// Resolves healing (Bishop's Consecration)
    /// </summary>
    public int ResolveHealing(BasePiece healer, BasePiece target)
    {
        var diceResult = DiceRoller.RollHealing();
        int healAmount = diceResult.FinalValue;

        target.Heal(healAmount);

        GD.Print($"[Heal] {healer.PieceType} heals {target.PieceType} for {healAmount}. " +
                 $"Target HP: {target.CurrentHp}/{target.MaxHp}");

        return healAmount;
    }
}
