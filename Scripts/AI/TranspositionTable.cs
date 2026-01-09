namespace Exchange.AI;

/// <summary>
/// Transposition table for caching evaluated board positions.
/// Uses depth-aware replacement and supports alpha-beta bound types.
/// Persists between moves for incremental reuse.
/// </summary>
public class TranspositionTable
{
    public enum NodeType : byte
    {
        Exact,      // Exact score
        LowerBound, // Score >= stored value (beta cutoff)
        UpperBound  // Score <= stored value (failed to raise alpha)
    }

    private struct Entry
    {
        public ulong Hash;
        public int Depth;
        public int Score;
        public NodeType Type;
        public SimulatedMove? BestMove;
        public byte Age; // For aging/replacement
    }

    // Power of 2 for fast modulo via bitwise AND
    private const int TableSize = 1 << 20; // ~1M entries (~32MB)
    private const int TableMask = TableSize - 1;

    private readonly Entry[] _table = new Entry[TableSize];
    private byte _currentAge = 0;

    /// <summary>
    /// Try to retrieve a cached evaluation.
    /// Returns true if we have a valid entry that can be used at this depth.
    /// </summary>
    public bool TryGet(ulong hash, int depth, int alpha, int beta, out int score)
    {
        int index = (int)(hash & TableMask);
        ref var entry = ref _table[index];

        score = 0;

        // Check if this entry matches our position
        if (entry.Hash != hash)
            return false;

        // Entry must be from search at least as deep as current
        if (entry.Depth < depth)
            return false;

        // Use the entry based on its node type
        switch (entry.Type)
        {
            case NodeType.Exact:
                score = entry.Score;
                return true;

            case NodeType.LowerBound:
                if (entry.Score >= beta)
                {
                    score = entry.Score;
                    return true;
                }
                break;

            case NodeType.UpperBound:
                if (entry.Score <= alpha)
                {
                    score = entry.Score;
                    return true;
                }
                break;
        }

        return false;
    }

    /// <summary>
    /// Get the best move from a previous search (for move ordering)
    /// </summary>
    public SimulatedMove? GetBestMove(ulong hash)
    {
        int index = (int)(hash & TableMask);
        ref var entry = ref _table[index];

        if (entry.Hash == hash)
            return entry.BestMove;

        return null;
    }

    /// <summary>
    /// Store an evaluation result
    /// </summary>
    public void Store(ulong hash, int depth, int score, NodeType type, SimulatedMove? bestMove)
    {
        int index = (int)(hash & TableMask);
        ref var entry = ref _table[index];

        // Replacement strategy:
        // - Always replace if empty (hash == 0)
        // - Replace if new entry is deeper
        // - Replace if entry is from older search
        // - Replace if entry is same depth but we have exact score
        bool shouldReplace =
            entry.Hash == 0 ||
            entry.Age != _currentAge ||
            entry.Depth <= depth ||
            (entry.Depth == depth && type == NodeType.Exact && entry.Type != NodeType.Exact);

        if (shouldReplace)
        {
            entry.Hash = hash;
            entry.Depth = depth;
            entry.Score = score;
            entry.Type = type;
            entry.BestMove = bestMove;
            entry.Age = _currentAge;
        }
    }

    /// <summary>
    /// Called after each move to age entries.
    /// Old entries remain valid for transpositions but can be replaced more easily.
    /// </summary>
    public void AgeEntries()
    {
        _currentAge++;
        // Wrap around is fine - entries from 255 moves ago can be replaced
    }

    /// <summary>
    /// Clear all entries (new game)
    /// </summary>
    public void Clear()
    {
        Array.Clear(_table, 0, _table.Length);
        _currentAge = 0;
    }

    /// <summary>
    /// Get statistics about table usage
    /// </summary>
    public (int used, int total) GetStats()
    {
        int used = 0;
        for (int i = 0; i < TableSize; i++)
        {
            if (_table[i].Hash != 0)
                used++;
        }
        return (used, TableSize);
    }
}
