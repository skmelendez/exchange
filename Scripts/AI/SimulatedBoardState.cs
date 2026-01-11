using Godot;
using Exchange.Core;
using Exchange.Board;
using Exchange.Pieces;

namespace Exchange.AI;

/// <summary>
/// Lightweight board state for AI simulation. No Godot Node overhead.
/// Supports make/undo moves and Zobrist hashing for transposition tables.
/// </summary>
public class SimulatedBoardState
{
    private readonly SimulatedPiece?[,] _board = new SimulatedPiece?[8, 8];
    private readonly List<SimulatedPiece> _pieces = new();
    private readonly Stack<MoveUndo> _undoStack = new();

    public ulong ZobristHash { get; private set; }

    public IReadOnlyList<SimulatedPiece> AllPieces => _pieces;

    /// <summary>
    /// Create simulation state from actual game board
    /// </summary>
    public static SimulatedBoardState FromGameBoard(GameBoard board)
    {
        var state = new SimulatedBoardState();

        foreach (var piece in board.PlayerPieces)
            state.AddPiece(SimulatedPiece.FromBasePiece(piece));

        foreach (var piece in board.EnemyPieces)
            state.AddPiece(SimulatedPiece.FromBasePiece(piece));

        state.RecalculateZobristHash();

        // Debug: Verify simulation matches real board
        VerifyBoardSync(state, board);

        // Debug: Log King HP
        var playerKing = state.GetKing(Team.Player);
        var enemyKing = state.GetKing(Team.Enemy);
        if (playerKing != null)
            GameLogger.Debug("AI-Init", $"Player King initialized: HP={playerKing.CurrentHp}/{playerKing.MaxHp}");
        if (enemyKing != null)
            GameLogger.Debug("AI-Init", $"Enemy King initialized: HP={enemyKing.CurrentHp}/{enemyKing.MaxHp}");

        return state;
    }

    private static void VerifyBoardSync(SimulatedBoardState simState, GameBoard realBoard)
    {
        // Check each position on the real board against simulation
        for (int x = 0; x < 8; x++)
        {
            for (int y = 0; y < 8; y++)
            {
                var pos = new Vector2I(x, y);
                var realPiece = realBoard.GetPieceAt(pos);
                var simPiece = simState.GetPieceAt(pos);

                if (realPiece != null && simPiece == null)
                {
                    GameLogger.Error("AI-Sync", $"MISSING in simulation: {realPiece.PieceType} at {pos.ToChessNotation()}");
                }
                else if (realPiece == null && simPiece != null)
                {
                    GameLogger.Error("AI-Sync", $"GHOST in simulation: {simPiece.PieceType} at {pos.ToChessNotation()}");
                }
                else if (realPiece != null && simPiece != null)
                {
                    if (realPiece.PieceType != simPiece.PieceType || realPiece.Team != simPiece.Team)
                    {
                        GameLogger.Error("AI-Sync", $"MISMATCH at {pos.ToChessNotation()}: Real={realPiece.Team} {realPiece.PieceType}, Sim={simPiece.Team} {simPiece.PieceType}");
                    }
                }
            }
        }
    }

    /// <summary>
    /// Deep copy for branching searches (if needed)
    /// </summary>
    public SimulatedBoardState Clone()
    {
        var clone = new SimulatedBoardState();
        foreach (var piece in _pieces)
        {
            clone.AddPiece(piece.Clone());
        }
        clone.ZobristHash = ZobristHash;
        return clone;
    }

    private void AddPiece(SimulatedPiece piece)
    {
        _pieces.Add(piece);
        _board[piece.Position.X, piece.Position.Y] = piece;
    }

    public SimulatedPiece? GetPieceAt(Vector2I pos)
    {
        if (!pos.IsOnBoard()) return null;
        return _board[pos.X, pos.Y];
    }

    public bool IsOccupied(Vector2I pos)
    {
        return GetPieceAt(pos) != null;
    }

    public bool IsOccupiedByTeam(Vector2I pos, Team team)
    {
        var piece = GetPieceAt(pos);
        return piece != null && piece.Team == team;
    }

    public IEnumerable<SimulatedPiece> GetPieces(Team team)
    {
        return _pieces.Where(p => p.Team == team && p.IsAlive);
    }

    public SimulatedPiece? GetKing(Team team)
    {
        return _pieces.FirstOrDefault(p => p.Team == team && p.PieceType == PieceType.King && p.IsAlive);
    }

    public bool HasKing(Team team)
    {
        return GetKing(team) != null;
    }

    #region Make/Undo Move

    public void MakeMove(SimulatedMove move)
    {
        var undo = new MoveUndo
        {
            Move = move,
            FromPos = move.FromPos,
            ToPos = move.ToPos,
            AttackPos = move.AttackPos,
            PreviousHash = ZobristHash
        };

        switch (move.MoveType)
        {
            case SimulatedMoveType.Move:
                MakeRegularMove(move, ref undo);
                break;

            case SimulatedMoveType.Attack:
                MakeAttackMove(move, ref undo);
                break;

            case SimulatedMoveType.MoveAndAttack:
                MakeMoveAndAttack(move, ref undo);
                break;

            case SimulatedMoveType.Ability:
                // TODO: Handle abilities in simulation if needed
                break;
        }

        _undoStack.Push(undo);
    }

    private void MakeRegularMove(SimulatedMove move, ref MoveUndo undo)
    {
        var piece = move.Piece;

        // Update Zobrist hash - remove piece from old position
        ZobristHash ^= ZobristKeys.GetPieceKey(piece.PieceType, piece.Team, move.FromPos);

        // Move piece on board
        _board[move.FromPos.X, move.FromPos.Y] = null;
        _board[move.ToPos.X, move.ToPos.Y] = piece;
        piece.Position = move.ToPos;

        // Update Zobrist hash - add piece to new position
        ZobristHash ^= ZobristKeys.GetPieceKey(piece.PieceType, piece.Team, move.ToPos);

        // Check for pawn promotion
        if (WouldPromote(piece, move.ToPos))
        {
            PromotePawn(piece, ref undo);
        }
    }

    private void MakeAttackMove(SimulatedMove move, ref MoveUndo undo)
    {
        var target = GetPieceAt(move.ToPos);
        if (target != null)
        {
            undo.TargetPreviousHp = target.CurrentHp;

            // Apply damage (removed debug logging for performance)
            target.CurrentHp -= move.DamageDealt;

            // Update hash for HP change
            ZobristHash ^= ZobristKeys.GetHpKey(target, undo.TargetPreviousHp);
            ZobristHash ^= ZobristKeys.GetHpKey(target, target.CurrentHp);

            // If killed, remove from board
            if (!target.IsAlive)
            {
                undo.CapturedPiece = target;
                _board[move.ToPos.X, move.ToPos.Y] = null;
                ZobristHash ^= ZobristKeys.GetPieceKey(target.PieceType, target.Team, move.ToPos);
            }
        }

        // Note: In this game, attacking doesn't move the attacker
        // If it did, we'd update position here too
    }

    /// <summary>
    /// Knight move+attack: move to ToPos, then attack piece at AttackPos
    /// </summary>
    private void MakeMoveAndAttack(SimulatedMove move, ref MoveUndo undo)
    {
        var piece = move.Piece;

        // Step 1: Move the piece
        ZobristHash ^= ZobristKeys.GetPieceKey(piece.PieceType, piece.Team, move.FromPos);
        _board[move.FromPos.X, move.FromPos.Y] = null;
        _board[move.ToPos.X, move.ToPos.Y] = piece;
        piece.Position = move.ToPos;
        ZobristHash ^= ZobristKeys.GetPieceKey(piece.PieceType, piece.Team, move.ToPos);

        // Step 2: Attack the target at AttackPos
        var target = GetPieceAt(move.AttackPos);
        if (target != null)
        {
            undo.TargetPreviousHp = target.CurrentHp;

            target.CurrentHp -= move.DamageDealt;

            ZobristHash ^= ZobristKeys.GetHpKey(target, undo.TargetPreviousHp);
            ZobristHash ^= ZobristKeys.GetHpKey(target, target.CurrentHp);

            if (!target.IsAlive)
            {
                undo.CapturedPiece = target;
                _board[move.AttackPos.X, move.AttackPos.Y] = null;
                ZobristHash ^= ZobristKeys.GetPieceKey(target.PieceType, target.Team, move.AttackPos);
            }
        }
    }

    public void UndoMove(SimulatedMove move)
    {
        if (_undoStack.Count == 0) return;

        var undo = _undoStack.Pop();

        switch (move.MoveType)
        {
            case SimulatedMoveType.Move:
                UndoRegularMove(undo);
                break;

            case SimulatedMoveType.Attack:
                UndoAttackMove(undo);
                break;

            case SimulatedMoveType.MoveAndAttack:
                UndoMoveAndAttack(undo);
                break;
        }

        ZobristHash = undo.PreviousHash;
    }

    private void UndoRegularMove(MoveUndo undo)
    {
        var piece = undo.Move.Piece;

        // Undo promotion first (if any)
        if (undo.WasPromotion)
        {
            UndoPromotion(piece, undo);
        }

        // Move piece back
        _board[undo.ToPos.X, undo.ToPos.Y] = null;
        _board[undo.FromPos.X, undo.FromPos.Y] = piece;
        piece.Position = undo.FromPos;
    }

    private void UndoAttackMove(MoveUndo undo)
    {
        var target = undo.CapturedPiece ?? GetPieceAt(undo.ToPos);
        if (target != null)
        {
            // Restore HP
            target.CurrentHp = undo.TargetPreviousHp;

            // If was killed, restore to board
            if (undo.CapturedPiece != null)
            {
                _board[undo.ToPos.X, undo.ToPos.Y] = undo.CapturedPiece;
            }
        }
    }

    private void UndoMoveAndAttack(MoveUndo undo)
    {
        var piece = undo.Move.Piece;

        // Undo attack first (restore target if killed)
        if (undo.CapturedPiece != null)
        {
            _board[undo.AttackPos.X, undo.AttackPos.Y] = undo.CapturedPiece;
            undo.CapturedPiece.CurrentHp = undo.TargetPreviousHp;
        }
        else
        {
            // Target wasn't killed, just restore HP
            var target = GetPieceAt(undo.AttackPos);
            if (target != null)
            {
                target.CurrentHp = undo.TargetPreviousHp;
            }
        }

        // Undo movement
        _board[undo.ToPos.X, undo.ToPos.Y] = null;
        _board[undo.FromPos.X, undo.FromPos.Y] = piece;
        piece.Position = undo.FromPos;
    }

    #endregion

    #region Zobrist Hashing

    private void RecalculateZobristHash()
    {
        ZobristHash = 0;
        foreach (var piece in _pieces)
        {
            if (piece.IsAlive)
            {
                ZobristHash ^= ZobristKeys.GetPieceKey(piece.PieceType, piece.Team, piece.Position);
                ZobristHash ^= ZobristKeys.GetHpKey(piece, piece.CurrentHp);
            }
        }
    }

    #endregion

    private struct MoveUndo
    {
        public SimulatedMove Move;
        public Vector2I FromPos;
        public Vector2I ToPos;
        public Vector2I AttackPos;  // For MoveAndAttack: where attack happened
        public SimulatedPiece? CapturedPiece;
        public int TargetPreviousHp;
        public ulong PreviousHash;
        // Promotion tracking
        public bool WasPromotion;
        public PieceType OriginalPieceType;
        public int OriginalMaxHp;
        public int OriginalBaseDamage;
    }

    #region Pawn Promotion Helpers

    /// <summary>
    /// Check if a pawn at the given position would promote
    /// </summary>
    public static bool WouldPromote(SimulatedPiece piece, Vector2I position)
    {
        if (piece.PieceType != PieceType.Pawn) return false;

        int promotionRow = piece.Team == Team.Player ? 7 : 0;
        return position.Y == promotionRow;
    }

    /// <summary>
    /// Promote a pawn in simulation (converts to Queen by default for AI)
    /// </summary>
    private void PromotePawn(SimulatedPiece pawn, ref MoveUndo undo)
    {
        if (pawn.PieceType != PieceType.Pawn) return;

        int promotionRow = pawn.Team == Team.Player ? 7 : 0;
        if (pawn.Position.Y != promotionRow) return;

        // Store original values for undo
        undo.WasPromotion = true;
        undo.OriginalPieceType = pawn.PieceType;
        undo.OriginalMaxHp = pawn.MaxHp;
        undo.OriginalBaseDamage = pawn.BaseDamage;

        // Remove old piece key from hash
        ZobristHash ^= ZobristKeys.GetPieceKey(pawn.PieceType, pawn.Team, pawn.Position);
        ZobristHash ^= ZobristKeys.GetHpKey(pawn, pawn.CurrentHp);

        // Calculate HP ratio to preserve relative health
        float hpRatio = (float)pawn.CurrentHp / pawn.MaxHp;

        // Promote to Queen (strongest piece, AI default choice)
        // Using stats from PieceData: Queen has 15 HP, 3 damage
        pawn.PromoteTo(PieceType.Queen, 15, 3, hpRatio);

        // Add new piece key to hash
        ZobristHash ^= ZobristKeys.GetPieceKey(pawn.PieceType, pawn.Team, pawn.Position);
        ZobristHash ^= ZobristKeys.GetHpKey(pawn, pawn.CurrentHp);

        GameLogger.Debug("AI-Sim", $"Pawn promoted to Queen at {pawn.Position.ToChessNotation()}");
    }

    /// <summary>
    /// Undo a pawn promotion
    /// </summary>
    private void UndoPromotion(SimulatedPiece piece, MoveUndo undo)
    {
        if (!undo.WasPromotion) return;

        // Remove current piece key
        ZobristHash ^= ZobristKeys.GetPieceKey(piece.PieceType, piece.Team, piece.Position);
        ZobristHash ^= ZobristKeys.GetHpKey(piece, piece.CurrentHp);

        // Restore original pawn stats
        float hpRatio = (float)piece.CurrentHp / piece.MaxHp;
        piece.DemoteTo(undo.OriginalPieceType, undo.OriginalMaxHp, undo.OriginalBaseDamage, hpRatio);

        // Add original piece key
        ZobristHash ^= ZobristKeys.GetPieceKey(piece.PieceType, piece.Team, piece.Position);
        ZobristHash ^= ZobristKeys.GetHpKey(piece, piece.CurrentHp);
    }

    #endregion
}

/// <summary>
/// Lightweight piece representation for AI simulation
/// </summary>
public class SimulatedPiece
{
    public PieceType PieceType { get; set; }
    public Team Team { get; init; }
    public Vector2I Position { get; set; }
    public int MaxHp { get; set; }
    public int CurrentHp { get; set; }
    public int BaseDamage { get; set; }

    public bool IsAlive => CurrentHp > 0;

    public static SimulatedPiece FromBasePiece(BasePiece piece)
    {
        return new SimulatedPiece
        {
            PieceType = piece.PieceType,
            Team = piece.Team,
            Position = piece.BoardPosition,
            MaxHp = piece.MaxHp,
            CurrentHp = piece.CurrentHp,
            BaseDamage = piece.BaseDamage
        };
    }

    public SimulatedPiece Clone()
    {
        return new SimulatedPiece
        {
            PieceType = PieceType,
            Team = Team,
            Position = Position,
            MaxHp = MaxHp,
            CurrentHp = CurrentHp,
            BaseDamage = BaseDamage
        };
    }

    /// <summary>
    /// Promote this pawn to a new piece type
    /// </summary>
    public void PromoteTo(PieceType newType, int newMaxHp, int newBaseDamage, float hpRatio)
    {
        PieceType = newType;
        MaxHp = newMaxHp;
        BaseDamage = newBaseDamage;
        CurrentHp = Math.Max(1, (int)(newMaxHp * hpRatio));
    }

    /// <summary>
    /// Demote back to original piece type (for undo)
    /// </summary>
    public void DemoteTo(PieceType originalType, int originalMaxHp, int originalBaseDamage, float hpRatio)
    {
        PieceType = originalType;
        MaxHp = originalMaxHp;
        BaseDamage = originalBaseDamage;
        CurrentHp = Math.Max(1, (int)(originalMaxHp * hpRatio));
    }
}

/// <summary>
/// Represents a move in simulation
/// </summary>
public struct SimulatedMove
{
    public SimulatedPiece Piece;
    public Vector2I FromPos;
    public Vector2I ToPos;
    public Vector2I AttackPos;  // For MoveAndAttack: where to attack after moving to ToPos
    public SimulatedMoveType MoveType;
    public SimulatedPiece? CapturedPiece;
    public int DamageDealt;
    public string Reason;

    public readonly string ToDebugString()
    {
        return MoveType switch
        {
            SimulatedMoveType.Move => $"{Piece.PieceType} {FromPos.ToChessNotation()}->{ToPos.ToChessNotation()}",
            SimulatedMoveType.Attack => $"{Piece.PieceType} x{ToPos.ToChessNotation()} ({DamageDealt}dmg)",
            SimulatedMoveType.MoveAndAttack => $"{Piece.PieceType} {FromPos.ToChessNotation()}->{ToPos.ToChessNotation()} x{AttackPos.ToChessNotation()} ({DamageDealt}dmg)",
            SimulatedMoveType.Ability => $"{Piece.PieceType} ability",
            _ => "Unknown"
        };
    }
}

public enum SimulatedMoveType
{
    Move,
    Attack,
    MoveAndAttack,  // Knight: move to ToPos, then attack AttackPos
    Ability
}

/// <summary>
/// Zobrist hash keys for board state fingerprinting.
/// Pre-generated random numbers ensure unique hashes for positions.
/// </summary>
public static class ZobristKeys
{
    // [pieceType][team][x][y]
    private static readonly ulong[,,,] PieceKeys = new ulong[6, 2, 8, 8];

    // HP changes - [pieceType][team][hp] (simplified, tracks HP 0-30)
    private static readonly ulong[,,] HpKeys = new ulong[6, 2, 31];

    // Side to move
    public static readonly ulong SideToMoveKey;

    static ZobristKeys()
    {
        var rng = new Random(unchecked((int)0xDEADBEEF)); // Fixed seed for reproducibility

        // Generate piece position keys
        for (int pt = 0; pt < 6; pt++)
        {
            for (int team = 0; team < 2; team++)
            {
                for (int x = 0; x < 8; x++)
                {
                    for (int y = 0; y < 8; y++)
                    {
                        PieceKeys[pt, team, x, y] = NextUlong(rng);
                    }
                }

                // HP keys
                for (int hp = 0; hp <= 30; hp++)
                {
                    HpKeys[pt, team, hp] = NextUlong(rng);
                }
            }
        }

        SideToMoveKey = NextUlong(rng);
    }

    public static ulong GetPieceKey(PieceType type, Team team, Vector2I pos)
    {
        int pt = (int)type;
        int t = team == Team.Player ? 0 : 1;
        return PieceKeys[pt, t, pos.X, pos.Y];
    }

    public static ulong GetHpKey(SimulatedPiece piece, int hp)
    {
        int pt = (int)piece.PieceType;
        int t = piece.Team == Team.Player ? 0 : 1;
        int clampedHp = Math.Clamp(hp, 0, 30);
        return HpKeys[pt, t, clampedHp];
    }

    private static ulong NextUlong(Random rng)
    {
        byte[] bytes = new byte[8];
        rng.NextBytes(bytes);
        return BitConverter.ToUInt64(bytes, 0);
    }
}
