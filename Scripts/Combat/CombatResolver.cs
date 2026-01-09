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

        // Boss Rule 1: Threat penalties are -2 during Room 1 Boss
        int threatPenalty = (_gameState.IsBossMatch && _gameState.CurrentRoom == 1) ? -2 : -1;

        // Roll dice with all modifiers
        var diceResult = DiceRoller.RollCombat(
            baseModifier: 0,
            enteredThreatZone: attacker.EnteredThreatZoneThisTurn,
            royalDecreeActive: royalDecreeActive,
            enemyKingWasThreatened: enemyKingThreatened,
            threatPenalty: threatPenalty
        );

        // Calculate total damage: Base Damage + Dice Roll
        int totalDamage = attacker.BaseDamage + diceResult.FinalValue;

        // Apply boss rules
        totalDamage = ApplyBossRules(totalDamage, attacker, defender, board);

        // Check for Interpose (Rook ability) - split damage with adjacent Rook
        var interposeRook = FindInterposeRook(defender, board);
        if (interposeRook != null)
        {
            int defenderDamage = totalDamage / 2;
            int rookDamage = totalDamage - defenderDamage; // Rook takes remainder

            defender.TakeDamage(defenderDamage);
            interposeRook.TakeDamage(rookDamage);

            GameLogger.Debug("Combat", $"INTERPOSE! {attacker.PieceType} attacks {defender.PieceType}: " +
                     $"Base {attacker.BaseDamage} + {diceResult.Breakdown} = {totalDamage} total. " +
                     $"Split: {defender.PieceType} takes {defenderDamage}, Rook takes {rookDamage}");
            GameLogger.Debug("Combat", $"{defender.PieceType} HP: {defender.CurrentHp}/{defender.MaxHp}, " +
                     $"Rook HP: {interposeRook.CurrentHp}/{interposeRook.MaxHp}");

            // Check if Rook died from interpose
            if (!interposeRook.IsAlive)
            {
                PieceDestroyed?.Invoke(interposeRook);
                GameLogger.Debug("Combat", "Rook destroyed from Interpose damage!");
            }
        }
        else
        {
            // Normal damage application
            defender.TakeDamage(totalDamage);

            GameLogger.Debug("Combat", $"{attacker.PieceType} attacks {defender.PieceType}: " +
                     $"Base {attacker.BaseDamage} + {diceResult.Breakdown} = {totalDamage} damage. " +
                     $"Defender HP: {defender.CurrentHp}/{defender.MaxHp}");
        }

        bool destroyed = !defender.IsAlive;
        if (destroyed)
        {
            PieceDestroyed?.Invoke(defender);
            GameLogger.Debug("Combat", $"{defender.PieceType} destroyed!");
        }

        var result = new CombatResult(attacker, defender, totalDamage, destroyed, diceResult);
        CombatResolved?.Invoke(result);

        return result;
    }

    /// <summary>
    /// Finds an adjacent allied Rook with Interpose active
    /// </summary>
    private RookPiece? FindInterposeRook(BasePiece defender, GameBoard board)
    {
        foreach (var dir in Vector2IExtensions.AllDirections)
        {
            var pos = defender.BoardPosition + dir;
            var piece = board.GetPieceAt(pos);
            if (piece is RookPiece rook && rook.Team == defender.Team && rook.InterposeActive)
            {
                return rook;
            }
        }
        return null;
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

        GameLogger.Debug("Heal", $"{healer.PieceType} heals {target.PieceType} for {healAmount}. " +
                 $"Target HP: {target.CurrentHp}/{target.MaxHp}");

        return healAmount;
    }
}
