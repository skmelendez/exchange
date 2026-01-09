using Godot;
using Exchange.Core;

namespace Exchange.Map;

/// <summary>
/// UI component that displays the act map with nodes and connections.
/// Renders left-to-right, allows clicking nodes to select them.
/// </summary>
public partial class MapUI : Control
{
    [Signal] public delegate void NodeClickedEventHandler(int nodeId);

    private RunManager? _runManager;
    private ActMap? _currentMap;

    // Layout constants
    private const float NodeSize = 40f;
    private const float HorizontalSpacing = 120f;
    private const float VerticalSpacing = 80f;
    private const float LeftMargin = 80f;
    private const float TopMargin = 100f;

    // Node colors by type
    private static readonly Dictionary<MapNodeType, Color> NodeColors = new()
    {
        { MapNodeType.Combat, new Color(0.7f, 0.3f, 0.3f) },    // Red
        { MapNodeType.Elite, new Color(0.8f, 0.6f, 0.2f) },     // Orange/Gold
        { MapNodeType.Boss, new Color(0.9f, 0.2f, 0.2f) },      // Bright Red
        { MapNodeType.Treasure, new Color(0.9f, 0.8f, 0.2f) },  // Yellow/Gold
        { MapNodeType.Shop, new Color(0.3f, 0.7f, 0.3f) },      // Green
        { MapNodeType.Event, new Color(0.5f, 0.5f, 0.8f) },     // Blue
        { MapNodeType.Start, new Color(0.5f, 0.5f, 0.5f) }      // Gray
    };

    // Node symbols
    private static readonly Dictionary<MapNodeType, string> NodeSymbols = new()
    {
        { MapNodeType.Combat, "!" },
        { MapNodeType.Elite, "E" },
        { MapNodeType.Boss, "B" },
        { MapNodeType.Treasure, "T" },
        { MapNodeType.Shop, "$" },
        { MapNodeType.Event, "?" },
        { MapNodeType.Start, "S" }
    };

    // Cached node positions for click detection
    private Dictionary<int, Rect2> _nodeRects = new();

    public void Initialize(RunManager runManager)
    {
        _runManager = runManager;
        _runManager.MapUpdated += OnMapUpdated;
    }

    public override void _Ready()
    {
        // Size will be set dynamically when map is loaded
    }

    private void OnMapUpdated()
    {
        _currentMap = _runManager?.CurrentActMap;
        QueueRedraw();
    }

    public void SetMap(ActMap? map)
    {
        _currentMap = map;

        // Calculate required size based on map dimensions
        if (map != null)
        {
            float requiredWidth = LeftMargin + (map.Config.Columns * HorizontalSpacing) + NodeSize;
            float requiredHeight = TopMargin + (map.Config.MaxRows * VerticalSpacing) + NodeSize;
            CustomMinimumSize = new Vector2(requiredWidth, requiredHeight);
        }

        QueueRedraw();
    }

    public override void _Draw()
    {
        if (_currentMap == null) return;

        _nodeRects.Clear();

        // Draw background
        DrawRect(new Rect2(Vector2.Zero, Size), new Color(0.1f, 0.1f, 0.15f));

        // Draw title
        var actTitle = _currentMap.ActNumber <= 3
            ? $"Act {_currentMap.ActNumber}"
            : "Final Boss";
        DrawString(ThemeDB.FallbackFont, new Vector2(20, 30), actTitle,
            HorizontalAlignment.Left, -1, 24, Colors.White);

        // Draw AI depth indicator
        var aiDepth = _runManager?.GetCurrentAiDepth() ?? 2;
        var isOverridden = _currentMap?.AiDepthOverride.HasValue ?? false;
        var aiText = isOverridden
            ? $"Enemy AI: {aiDepth}-ply lookahead (custom)"
            : $"Enemy AI: {aiDepth}-ply lookahead";
        var aiColor = isOverridden ? new Color(0.9f, 0.6f, 0.2f) : Colors.Gray;
        DrawString(ThemeDB.FallbackFont, new Vector2(20, 55),
            aiText,
            HorizontalAlignment.Left, -1, 14, aiColor);

        // Draw connections first (below nodes)
        DrawConnections();

        // Draw nodes
        DrawNodes();
    }

    private void DrawConnections()
    {
        if (_currentMap == null) return;

        foreach (var node in _currentMap.Nodes)
        {
            var startPos = GetNodeScreenPosition(node);

            foreach (var targetId in node.OutgoingConnections)
            {
                var target = _currentMap.GetNodeById(targetId);
                if (target == null) continue;

                var endPos = GetNodeScreenPosition(target);

                // Determine line color based on state
                Color lineColor;
                float lineWidth;

                if (node.IsVisited && target.IsAccessible)
                {
                    // Active path
                    lineColor = new Color(0.4f, 0.8f, 0.4f, 0.9f);
                    lineWidth = 3f;
                }
                else if (node.IsVisited || target.IsVisited)
                {
                    // Partially traveled
                    lineColor = new Color(0.6f, 0.6f, 0.6f, 0.7f);
                    lineWidth = 2f;
                }
                else
                {
                    // Unexplored
                    lineColor = new Color(0.4f, 0.4f, 0.4f, 0.5f);
                    lineWidth = 1.5f;
                }

                // Draw bezier curve for nicer look
                DrawBezierConnection(startPos, endPos, lineColor, lineWidth);
            }
        }
    }

    private void DrawBezierConnection(Vector2 start, Vector2 end, Color color, float width)
    {
        // Simple bezier with control points
        float midX = (start.X + end.X) / 2;
        var ctrl1 = new Vector2(midX, start.Y);
        var ctrl2 = new Vector2(midX, end.Y);

        // Draw as line segments
        const int segments = 12;
        var prevPoint = start;

        for (int i = 1; i <= segments; i++)
        {
            float t = i / (float)segments;
            var point = CubicBezier(start, ctrl1, ctrl2, end, t);
            DrawLine(prevPoint, point, color, width);
            prevPoint = point;
        }
    }

    private Vector2 CubicBezier(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t)
    {
        float u = 1 - t;
        float tt = t * t;
        float uu = u * u;
        float uuu = uu * u;
        float ttt = tt * t;

        return uuu * p0 + 3 * uu * t * p1 + 3 * u * tt * p2 + ttt * p3;
    }

    private void DrawNodes()
    {
        if (_currentMap == null) return;

        foreach (var node in _currentMap.Nodes)
        {
            DrawNode(node);
        }
    }

    private void DrawNode(MapNode node)
    {
        var pos = GetNodeScreenPosition(node);
        var rect = new Rect2(pos.X - NodeSize / 2, pos.Y - NodeSize / 2, NodeSize, NodeSize);

        // Store for click detection
        _nodeRects[node.Id] = rect;

        // Determine colors based on state
        Color bgColor = NodeColors.GetValueOrDefault(node.NodeType, Colors.Gray);
        Color borderColor;
        float borderWidth;
        string symbol = NodeSymbols.GetValueOrDefault(node.NodeType, "?");
        Color textColor = Colors.White;

        if (node.IsCurrentNode)
        {
            borderColor = Colors.White;
            borderWidth = 4f;
        }
        else if (node.IsCompleted)
        {
            // Completed nodes: red background with "W" for combat wins
            bgColor = new Color(0.8f, 0.2f, 0.2f); // Bright red
            borderColor = new Color(0.6f, 0.1f, 0.1f);
            borderWidth = 2f;
            if (node.IsCombatNode)
            {
                symbol = "W";
            }
            textColor = Colors.White;
        }
        else if (node.IsAccessible)
        {
            borderColor = new Color(0.4f, 1f, 0.4f);
            borderWidth = 3f;
            // Pulse effect could be added with animation
        }
        else if (node.IsVisited)
        {
            bgColor = bgColor.Darkened(0.4f);
            borderColor = Colors.DarkGray;
            borderWidth = 2f;
            textColor = Colors.DarkGray;
        }
        else
        {
            bgColor = bgColor.Darkened(0.2f);
            borderColor = new Color(0.3f, 0.3f, 0.3f);
            borderWidth = 1.5f;
        }

        // Draw node background (circle approximation with rounded rect)
        DrawRect(rect, bgColor);

        // Draw border
        DrawRect(rect, borderColor, false, borderWidth);

        // Draw symbol
        var textPos = pos + new Vector2(-6, 6);  // Center-ish
        DrawString(ThemeDB.FallbackFont, textPos, symbol,
            HorizontalAlignment.Left, -1, 16, textColor);

        // Draw "current" indicator
        if (node.IsCurrentNode)
        {
            DrawCircle(pos, NodeSize / 2 + 8, new Color(1, 1, 1, 0.3f));
        }
    }

    private Vector2 GetNodeScreenPosition(MapNode node)
    {
        float x = LeftMargin + node.Position.Column * HorizontalSpacing;
        float y = TopMargin + node.Position.Row * VerticalSpacing + (NodeSize / 2);
        return new Vector2(x, y);
    }

    public override void _GuiInput(InputEvent @event)
    {
        if (@event is InputEventMouseButton mb && mb.Pressed && mb.ButtonIndex == MouseButton.Left)
        {
            var clickPos = mb.Position;

            foreach (var (nodeId, rect) in _nodeRects)
            {
                // Expand rect slightly for easier clicking
                var expandedRect = rect.Grow(5);
                if (expandedRect.HasPoint(clickPos))
                {
                    GameLogger.Debug("MapUI", $"Node {nodeId} clicked");
                    EmitSignal(SignalName.NodeClicked, nodeId);
                    return;
                }
            }
        }
    }
}
