"""
Web Dashboard for EXCHANGE Training

A simple web-based game viewer that lets you:
- Watch games move by move
- Browse saved replays
- View training progress

Usage:
    python -m src.dashboard
    # Then open http://localhost:5000

    # Or specify port:
    python -m src.dashboard --port 8080
"""

from __future__ import annotations

import json
import os
import io
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import threading
import time
from datetime import datetime
import subprocess

from .game_state import GameState, PieceType, Team
from .game_simulator import GameSimulator, Move, MoveType


# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>EXCHANGE Training Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #00d4ff; margin-bottom: 20px; }
        h2 { color: #888; font-size: 14px; margin-bottom: 10px; text-transform: uppercase; }

        .grid { display: grid; grid-template-columns: 300px 1fr; gap: 20px; }

        .sidebar {
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            max-height: 80vh;
            overflow-y: auto;
        }

        .game-list { list-style: none; }
        .game-list li {
            padding: 10px;
            margin: 5px 0;
            background: #1a1a2e;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .game-list li:hover { background: #0f3460; }
        .game-list .winner { font-size: 12px; color: #888; }
        .game-list li.unwatched { background: #5c5c1a; }
        .game-list li.unwatched:hover { background: #6e6e22; }
        .game-list li.watched { background: #1a4d1a; }
        .game-list li.watched:hover { background: #226622; }
        .game-list .watch-status { font-size: 11px; font-weight: bold; margin-top: 4px; }
        .game-list .watch-status.unwatched { color: #ffeb3b; }
        .game-list .watch-status.watched { color: #4caf50; }
        /* Active (selected) state - must come AFTER unwatched/watched to override */
        .game-list li.active { background: #ffffff !important; color: #000 !important; font-weight: bold; }
        .game-list li.active .winner { color: #333; }
        .game-list li.active .watch-status { color: #333; }

        .main-panel {
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
        }

        .board-container { text-align: center; margin-bottom: 20px; }

        .board-area {
            display: flex;
            align-items: stretch;
            justify-content: center;
            gap: 10px;
        }

        .eval-bar {
            width: 60px;
            display: flex;
            flex-direction: column;
            border-radius: 4px;
            padding: 8px 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
        }
        .eval-bar.white-bar {
            background: #f0f0f0;
            color: #000;
        }
        .eval-bar.black-bar {
            background: #1a1a1a;
            color: #fff;
        }
        .eval-bar .bar-title {
            font-weight: bold;
            text-align: center;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid;
        }
        .eval-bar.white-bar .bar-title { border-color: #ccc; }
        .eval-bar.black-bar .bar-title { border-color: #444; }
        .eval-bar .bar-total {
            font-size: 16px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 8px;
        }
        .eval-bar .piece-score {
            display: flex;
            justify-content: space-between;
            padding: 2px 0;
            font-size: 10px;
        }
        .eval-bar .piece-score .piece-name { font-weight: bold; }

        .advantage-bar {
            width: 100%;
            height: 20px;
            background: linear-gradient(to right, #fff 50%, #000 50%);
            border-radius: 4px;
            margin: 10px 0;
            position: relative;
            overflow: hidden;
        }
        .advantage-fill {
            position: absolute;
            top: 0;
            height: 100%;
            transition: all 0.3s;
        }
        .advantage-fill.white-advantage {
            left: 50%;
            background: #fff;
        }
        .advantage-fill.black-advantage {
            right: 50%;
            background: #000;
        }
        .advantage-label {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 11px;
            font-weight: bold;
            color: #888;
            text-shadow: 0 0 3px #fff, 0 0 3px #000;
        }

        .board {
            display: inline-grid;
            grid-template-columns: repeat(8, 50px);
            grid-template-rows: repeat(8, 50px);
            gap: 1px;
            background: #333;
            padding: 1px;
            border-radius: 4px;
        }

        .square {
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            position: relative;
        }
        .square.light { background: #b58863; }
        .square.dark { background: #f0d9b5; }
        .square.origin { box-shadow: inset 0 0 0 3px #00d4ff; }
        .square.destination { box-shadow: inset 0 0 0 3px #44ff44; }
        .square.attack { box-shadow: inset 0 0 0 3px #ff4444; }

        .board-wrapper {
            position: relative;
            display: inline-block;
        }
        .move-overlay {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }
        .move-line {
            stroke-width: 3;
            stroke-linecap: round;
        }
        .move-line.movement { stroke: #44ff44; }
        .move-line.attack { stroke: #ff4444; }

        .piece { cursor: default; font-weight: bold; font-family: monospace; }
        .piece.white { color: #ffffff; text-shadow: 1px 1px 2px #000, -1px -1px 2px #000; }
        .piece.black { color: #000000; text-shadow: 1px 1px 1px #888; }

        .hp-bar {
            position: absolute;
            bottom: 2px;
            left: 2px;
            right: 2px;
            height: 4px;
            background: #333;
            border-radius: 2px;
        }
        .hp-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s;
        }
        .hp-fill.healthy { background: #4caf50; }
        .hp-fill.damaged { background: #ff9800; }
        .hp-fill.critical { background: #f44336; }

        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 20px;
        }

        button {
            background: #00d4ff;
            border: none;
            color: #000;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.2s;
        }
        button:hover { background: #00a8cc; }
        button:disabled { background: #555; cursor: not-allowed; }

        .move-info {
            background: #1a1a2e;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }
        .move-info .turn { color: #00d4ff; font-size: 18px; }
        .move-info .move { font-size: 24px; margin: 10px 0; }
        .move-info .damage { color: #ff4444; }
        .move-info .ability-move { color: #ffd700; font-weight: bold; }

        .piece-list {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 15px;
            font-size: 14px;
        }
        .piece-list .white { color: #4caf50; }
        .piece-list .black { color: #f44336; }

        .no-game {
            text-align: center;
            padding: 100px 20px;
            color: #666;
        }

        #live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #4caf50;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .autoplay-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            justify-content: center;
            margin-top: 10px;
        }
        .autoplay-controls label { color: #888; }
        .autoplay-controls input[type="range"] { width: 100px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>EXCHANGE Training Dashboard</h1>

        <div class="grid">
            <div class="sidebar">
                <h2>Saved Games</h2>
                <button onclick="watchLiveGame()" style="width: 100%; margin-bottom: 15px;">
                    <span id="live-indicator" style="display: none;"></span>
                    Watch New Game
                </button>

                <div style="margin-bottom: 15px; padding: 10px; background: #1a1a2e; border-radius: 4px;">
                    <h2 style="margin-bottom: 8px;">Analysis Report</h2>
                    <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                        <button onclick="downloadAnalysis(50)" style="flex: 1; min-width: 60px; padding: 6px 8px; font-size: 12px;">50</button>
                        <button onclick="downloadAnalysis(100)" style="flex: 1; min-width: 60px; padding: 6px 8px; font-size: 12px;">100</button>
                        <button onclick="downloadAnalysis(200)" style="flex: 1; min-width: 60px; padding: 6px 8px; font-size: 12px;">200</button>
                        <button onclick="downloadAnalysis(500)" style="flex: 1; min-width: 60px; padding: 6px 8px; font-size: 12px;">500</button>
                        <button onclick="downloadAnalysis(1000)" style="flex: 1; min-width: 60px; padding: 6px 8px; font-size: 12px;">1000</button>
                    </div>
                    <div id="analysis-status" style="font-size: 11px; color: #888; margin-top: 5px; text-align: center;"></div>
                </div>
                <div style="margin-bottom: 10px;">
                    <label style="color: #888; font-size: 12px;">Iteration:</label>
                    <select id="iteration-filter" onchange="onIterationChange()" style="width: 100%; padding: 8px; border-radius: 4px; background: #1a1a2e; color: #fff; border: 1px solid #333; margin-top: 5px;">
                        <option value="latest">Latest</option>
                    </select>
                </div>
                <div style="margin-bottom: 15px;">
                    <label style="color: #888; font-size: 12px;">Filter by outcome:</label>
                    <select id="outcome-filter" onchange="onFilterChange()" style="width: 100%; padding: 8px; border-radius: 4px; background: #1a1a2e; color: #fff; border: 1px solid #333; margin-top: 5px;">
                        <option value="all">All Games</option>
                        <option value="decisive">Decisive (No Draws)</option>
                        <option value="white">White Wins</option>
                        <option value="black">Black Wins</option>
                        <option value="draw">Draws</option>
                    </select>
                </div>
                <ul class="game-list" id="game-list">
                    <li class="no-games">Loading games...</li>
                </ul>
            </div>

            <div class="main-panel">
                <div id="game-view" class="no-game">
                    <h2>Select a game or watch a new one</h2>
                    <p>Click "Watch New Game" to see the AI play, or select a saved replay.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Piece type mapping: 0=King, 1=Queen, 2=Rook, 3=Bishop, 4=Knight, 5=Pawn
        const PIECE_LETTERS = ['K', 'Q', 'R', 'B', 'N', 'P'];
        const PIECE_NAMES = ['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn'];
        // Material values (King=0 for material calc, but infinite strategic value)
        const PIECE_VALUES = [0, 9, 5, 3, 3, 1];
        // Max HP per piece type
        const PIECE_MAX_HP = [25, 10, 13, 10, 11, 7];

        // Convert string piece type to number (handles both formats)
        const PIECE_NAME_TO_ID = { 'King': 0, 'Queen': 1, 'Rook': 2, 'Bishop': 3, 'Knight': 4, 'Pawn': 5 };
        const TEAM_NAME_TO_ID = { 'White': 0, 'Black': 1 };

        function normalizePieceType(pt) {
            if (typeof pt === 'number') return pt;
            if (typeof pt === 'string') return PIECE_NAME_TO_ID[pt] ?? -1;
            return -1;
        }
        function normalizeTeam(t) {
            if (typeof t === 'number') return t;
            if (typeof t === 'string') return TEAM_NAME_TO_ID[t] ?? 0;
            return 0;
        }

        let currentGame = null;
        let currentGamePath = null;
        let currentMoveIndex = -1;
        let autoplayInterval = null;
        let autoplaySpeed = parseInt(localStorage.getItem('autoplaySpeed')) || 500;
        let gameListRefreshInterval = null;
        const REFRESH_INTERVAL_MS = 5000; // Check for new games every 5 seconds
        let availableIterations = []; // Populated from API

        // URL Query String helpers
        function getQueryParams() {
            const params = new URLSearchParams(window.location.search);
            return {
                iteration: params.get('iteration') || 'latest',
                outcome: params.get('outcome') || 'all'
            };
        }

        function setQueryParams(iteration, outcome) {
            const params = new URLSearchParams();
            if (iteration !== 'latest') params.set('iteration', iteration);
            if (outcome !== 'all') params.set('outcome', outcome);
            const newUrl = params.toString() ? `?${params}` : window.location.pathname;
            history.replaceState(null, '', newUrl);
        }

        function onIterationChange() {
            const iteration = document.getElementById('iteration-filter').value;
            const outcome = document.getElementById('outcome-filter').value;
            setQueryParams(iteration, outcome);
            loadGameList();
        }

        function onFilterChange() {
            const iteration = document.getElementById('iteration-filter').value;
            const outcome = document.getElementById('outcome-filter').value;
            setQueryParams(iteration, outcome);
            loadGameList();
        }

        function populateIterationDropdown(iterations, currentIteration) {
            const select = document.getElementById('iteration-filter');
            const params = getQueryParams();

            // Build options
            let options = '<option value="latest">Latest Iteration</option>';
            options += '<option value="all">All Iterations</option>';
            iterations.forEach(iter => {
                options += `<option value="${iter}">Iteration ${iter}</option>`;
            });

            select.innerHTML = options;

            // Set value from URL params
            select.value = params.iteration;
        }

        function syncDropdownsFromUrl() {
            const params = getQueryParams();
            document.getElementById('iteration-filter').value = params.iteration;
            document.getElementById('outcome-filter').value = params.outcome;
        }

        // Watched games tracking - uses "<iter>_<gamenum>" format
        function getWatchedGames() {
            const stored = localStorage.getItem('watchedGames');
            return stored ? JSON.parse(stored) : [];
        }

        function markGameAsWatched(gameId) {
            // gameId format: "<iter>_<gamenum>" e.g., "50_0001"
            const watched = getWatchedGames();
            if (!watched.includes(gameId)) {
                watched.push(gameId);
                localStorage.setItem('watchedGames', JSON.stringify(watched));
            }
        }

        function isGameWatched(gameId) {
            // gameId format: "<iter>_<gamenum>" e.g., "50_0001"
            return getWatchedGames().includes(gameId);
        }

        function saveAutoplaySpeed(speed) {
            localStorage.setItem('autoplaySpeed', speed.toString());
        }

        async function loadGameList() {
            const params = getQueryParams();
            const url = `/api/games?iteration=${params.iteration}&outcome=${params.outcome}`;
            const response = await fetch(url);
            const data = await response.json();

            availableIterations = data.available_iterations;
            populateIterationDropdown(data.available_iterations, data.current_iteration);
            syncDropdownsFromUrl();
            renderGameList(data.games, data.total_count);
        }

        function renderGameList(games, totalCount) {
            const list = document.getElementById('game-list');

            if (!games || games.length === 0) {
                list.innerHTML = totalCount === 0
                    ? '<li class="no-games">No saved games yet. Click "Watch New Game"!</li>'
                    : '<li class="no-games">No matching games found.</li>';
                return;
            }

            list.innerHTML = games.map(g => {
                // Use id format "<iter>_<gamenum>" for watched tracking
                const watched = isGameWatched(g.id);
                const watchClass = watched ? 'watched' : 'unwatched';
                const watchLabel = watched ? 'Watched' : 'New';
                const isActive = currentGamePath === g.path ? 'active' : '';
                // Convert moves to turns (1 turn = White + Black both moved)
                const actualTurns = Math.ceil(g.moves / 2);
                // Winner badge colors
                const winnerColor = g.winner === 'WHITE' ? '#fff' : g.winner === 'BLACK' ? '#333' : '#888';
                const winnerBg = g.winner === 'WHITE' ? '#444' : g.winner === 'BLACK' ? '#ccc' : '#555';
                const gameLabel = g.iteration >= 0 ? `Game ${g.game_num.toString().padStart(2, '0')}` : g.id;
                return `
                    <li class="${watchClass} ${isActive}" onclick="loadGame('${g.path}', '${g.id}')">
                        <span style="display: flex; justify-content: space-between; align-items: center;">
                            <span>${gameLabel}</span>
                            <span style="background: ${winnerBg}; color: ${winnerColor}; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold;">${g.winner}</span>
                        </span>
                        <div class="winner">${actualTurns} turns (${g.moves} moves)</div>
                        <div class="watch-status ${watchClass}">&lt;${watchLabel}&gt;</div>
                    </li>
                `;
            }).join('');
        }

        async function loadGame(path, gameId) {
            stopAutoplay();
            const response = await fetch('/api/game?path=' + encodeURIComponent(path));
            currentGame = await response.json();
            currentGamePath = path;
            currentMoveIndex = -1;

            // Mark as watched using "<iter>_<gamenum>" format and refresh list
            if (gameId) {
                markGameAsWatched(gameId);
            }
            renderGame();
            loadGameList(); // Refresh to show updated watched status
        }

        async function watchLiveGame() {
            stopAutoplay();
            document.getElementById('live-indicator').style.display = 'inline-block';

            const response = await fetch('/api/play');
            currentGame = await response.json();
            currentMoveIndex = -1;

            document.getElementById('live-indicator').style.display = 'none';
            renderGame();
            loadGameList(); // Refresh list with new game
        }

        function renderGame() {
            if (!currentGame) return;

            const view = document.getElementById('game-view');
            view.innerHTML = `
                <div class="board-container">
                    <div class="advantage-bar" id="advantage-bar">
                        <div class="advantage-fill" id="advantage-fill"></div>
                        <div class="advantage-label" id="advantage-label">EVEN</div>
                    </div>
                    <div class="board-area">
                        <div class="eval-bar white-bar" id="white-eval">
                            <div class="bar-title">WHITE</div>
                            <div class="bar-total" id="white-total">0.0</div>
                            <div id="white-pieces"></div>
                        </div>
                        <div class="board-wrapper">
                            <div class="board" id="board"></div>
                            <svg class="move-overlay" id="move-overlay" width="409" height="409"></svg>
                        </div>
                        <div class="eval-bar black-bar" id="black-eval">
                            <div class="bar-title">BLACK</div>
                            <div class="bar-total" id="black-total">0.0</div>
                            <div id="black-pieces"></div>
                        </div>
                    </div>
                </div>

                <div class="scrubber-container" style="margin-bottom: 15px; padding: 0 10px;">
                    <input type="range" id="move-scrubber" min="-1" max="0" value="-1"
                           oninput="scrubToMove(this.value)"
                           style="width: 100%; cursor: pointer;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #888; margin-top: 5px;">
                        <span>Start</span>
                        <span id="scrubber-position">Game Start</span>
                        <span id="scrubber-total">End</span>
                    </div>
                </div>

                <div class="controls">
                    <button onclick="firstMove()">|&lt; First</button>
                    <button onclick="prevMove()">&lt; Prev</button>
                    <button onclick="nextMove()">Next &gt;</button>
                    <button onclick="lastMove()">Last &gt;|</button>
                </div>

                <div class="autoplay-controls">
                    <button onclick="toggleAutoplay()" id="autoplay-btn">Play</button>
                    <label>Speed:</label>
                    <input type="range" id="speed-slider" min="100" max="2000" value="${2100 - autoplaySpeed}" oninput="updateAutoplaySpeed(this.value)">
                </div>

                <div class="move-info" id="move-info"></div>
                <div class="piece-list" id="piece-list"></div>
            `;

            initScrubber();
            renderPosition();
        }

        function renderPosition() {
            if (!currentGame) return;

            // Reconstruct state AFTER the current move (currentMoveIndex + 1 moves applied)
            // -1 = initial state (0 moves), 0 = after first move, etc.
            const state = reconstructState(currentMoveIndex + 1);
            const board = document.getElementById('board');
            const moveInfo = document.getElementById('move-info');
            const pieceList = document.getElementById('piece-list');
            const overlay = document.getElementById('move-overlay');

            // Get the move that was just made (if any)
            const move = currentMoveIndex >= 0 ? currentGame.moves[currentMoveIndex] : null;

            // Helper functions - handles both numeric and string formats
            const getPieceHp = (p) => p && typeof p.currentHp === 'number' ? p.currentHp : 0;
            const getPieceType = (p) => p ? normalizePieceType(p.pieceType) : -1;
            const getPieceTeam = (p) => p ? normalizeTeam(p.team) : 0;
            const getPieceMaxHp = (p) => p && typeof p.maxHp === 'number' ? p.maxHp : (PIECE_MAX_HP[getPieceType(p)] || 10);
            const getPieceWasPawn = (p) => p && p.wasPawn === true;

            // Render board
            let html = '';
            for (let y = 7; y >= 0; y--) {
                for (let x = 0; x < 8; x++) {
                    const isLight = (x + y) % 2 === 1;
                    const piece = state.pieces.find(p => p.x === x && p.y === y && getPieceHp(p) > 0);

                    let classes = 'square ' + (isLight ? 'light' : 'dark');

                    // Highlight the move that was just made
                    if (move) {
                        if (move.from[0] === x && move.from[1] === y) classes += ' origin';
                        // For destination, check if piece actually moved
                        const pieceMoved = move.move_type === 0 || move.move_type === 2 ||
                            (move.move_type === 3 && (move.to[0] !== move.from[0] || move.to[1] !== move.from[1]));
                        if (pieceMoved && move.to[0] === x && move.to[1] === y) classes += ' destination';
                        // Attack target
                        if (move.attack && move.attack[0] === x && move.attack[1] === y) classes += ' attack';
                        // For regular attacks (move_type 1), the 'to' is the attack target
                        if (move.move_type === 1 && move.to[0] === x && move.to[1] === y) classes += ' attack';
                    }

                    html += `<div class="${classes}">`;
                    if (piece) {
                        // Show promoted pieces as "PQ", "PN", etc.
                        const pieceType = getPieceType(piece);
                        const currentHp = getPieceHp(piece);
                        const maxHp = getPieceMaxHp(piece);
                        const wasPawn = getPieceWasPawn(piece);

                        let symbol = (pieceType >= 0 && pieceType < PIECE_LETTERS.length) ? PIECE_LETTERS[pieceType] : '?';
                        if (wasPawn && pieceType !== 5) {
                            symbol = 'P' + symbol;  // e.g., "PQ" for promoted Queen
                        }
                        const colorClass = getPieceTeam(piece) === 0 ? 'white' : 'black';
                        const hpPct = maxHp > 0 ? (currentHp / maxHp) * 100 : 0;
                        const hpClass = hpPct > 60 ? 'healthy' : hpPct > 30 ? 'damaged' : 'critical';

                        html += `<span class="piece ${colorClass}">${symbol}</span>`;
                        html += `<div class="hp-bar"><div class="hp-fill ${hpClass}" style="width: ${hpPct}%"></div></div>`;
                    }
                    html += '</div>';
                }
            }
            board.innerHTML = html;

            // Draw movement and attack lines on SVG overlay
            let svgContent = '';
            if (move) {
                const SQUARE_SIZE = 50;
                const GAP = 1;
                const PADDING = 1;
                // Helper to get center of square (board is rendered with y=7 at top)
                // Account for 1px gaps between squares and 1px padding
                const getCenter = (x, y) => ({
                    x: PADDING + x * (SQUARE_SIZE + GAP) + SQUARE_SIZE / 2,
                    y: PADDING + (7 - y) * (SQUARE_SIZE + GAP) + SQUARE_SIZE / 2
                });

                const fromCenter = getCenter(move.from[0], move.from[1]);

                // Draw movement line (green) if piece moved
                const pieceMoved = move.move_type === 0 || move.move_type === 2 ||
                    (move.move_type === 3 && (move.to[0] !== move.from[0] || move.to[1] !== move.from[1]));
                if (pieceMoved) {
                    const toCenter = getCenter(move.to[0], move.to[1]);
                    svgContent += `<line class="move-line movement" x1="${fromCenter.x}" y1="${fromCenter.y}" x2="${toCenter.x}" y2="${toCenter.y}" />`;
                }

                // Draw attack line (red) if there was an attack
                if (move.damage > 0) {
                    // Check if attack position is valid (not [-1,-1] sentinel)
                    const hasValidAttack = move.attack &&
                        Array.isArray(move.attack) &&
                        move.attack[0] >= 0 && move.attack[1] >= 0;

                    let attackTarget;
                    if (hasValidAttack) {
                        // Explicit attack position (Knight combo, Overextend, Skirmish)
                        attackTarget = getCenter(move.attack[0], move.attack[1]);
                    } else if (move.move_type === 1) {
                        // Regular ATTACK - target is at 'to' position
                        attackTarget = getCenter(move.to[0], move.to[1]);
                    }

                    if (attackTarget) {
                        // Draw from piece's current position (after move if it moved)
                        const piecePos = pieceMoved ? getCenter(move.to[0], move.to[1]) : fromCenter;
                        svgContent += `<line class="move-line attack" x1="${piecePos.x}" y1="${piecePos.y}" x2="${attackTarget.x}" y2="${attackTarget.y}" />`;
                    }
                }
            }
            overlay.innerHTML = svgContent;

            // Render move info
            if (move) {
                const pieceNames = ['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn'];
                const abilityNames = ['Royal Decree', 'Overextend', 'Interpose', 'Consecration', 'Skirmish', 'Advance'];
                const teamName = move.team === 0 ? 'WHITE' : 'BLACK';
                const pieceName = pieceNames[move.piece_type];
                const from = String.fromCharCode(97 + move.from[0]) + (move.from[1] + 1);
                const to = String.fromCharCode(97 + move.to[0]) + (move.to[1] + 1);

                let moveStr;
                // Check if this is an ability move (move_type 3)
                if (move.move_type === 3 && move.ability_id !== null && move.ability_id !== undefined) {
                    const abilityName = abilityNames[move.ability_id] || 'ABILITY';
                    moveStr = `<span class="ability-move">${pieceName} ${abilityName}</span>`;
                    if (move.ability_target) {
                        const target = String.fromCharCode(97 + move.ability_target[0]) + (move.ability_target[1] + 1);
                        moveStr += ` -> ${target}`;
                    } else if (move.to && (move.to[0] !== move.from[0] || move.to[1] !== move.from[1])) {
                        moveStr += ` -> ${to}`;
                    }
                } else {
                    moveStr = `${pieceName} ${from} -> ${to}`;
                }
                if (move.damage > 0) moveStr += ` <span class="damage">(${move.damage} damage!)</span>`;

                // Calculate actual turn (1 turn = both sides moved)
                const actualTurn = Math.ceil((currentMoveIndex + 2) / 2);
                const totalTurns = Math.ceil(currentGame.moves.length / 2);

                moveInfo.innerHTML = `
                    <div class="turn">Turn ${actualTurn} - ${teamName}</div>
                    <div class="move">${moveStr}</div>
                    <div>Move ${currentMoveIndex + 1} of ${currentGame.moves.length}</div>
                `;
            } else {
                moveInfo.innerHTML = `<div class="turn">Game Start</div><div class="move">Initial Position</div>`;
            }

            // Render piece lists
            const whitePieces = state.pieces.filter(p => getPieceTeam(p) === 0 && getPieceHp(p) > 0);
            const blackPieces = state.pieces.filter(p => getPieceTeam(p) === 1 && getPieceHp(p) > 0);

            // Helper to get piece symbol (with P prefix for promoted)
            const getPieceSymbol = (p) => {
                const pType = getPieceType(p);
                const wasPawn = getPieceWasPawn(p);
                let sym = (pType >= 0 && pType < PIECE_LETTERS.length) ? PIECE_LETTERS[pType] : '?';
                if (wasPawn && pType !== 5) sym = 'P' + sym;
                return sym;
            };

            pieceList.innerHTML = `
                <span class="white">WHITE: ${whitePieces.map(p => getPieceSymbol(p) + '(' + getPieceHp(p) + ')').join(' ')}</span>
                <span class="black">BLACK: ${blackPieces.map(p => getPieceSymbol(p) + '(' + getPieceHp(p) + ')').join(' ')}</span>
            `;

            // Calculate and render eval bars
            const calcPieceScore = (p) => {
                const pType = getPieceType(p);
                const hp = getPieceHp(p);
                const maxHp = getPieceMaxHp(p);
                const materialValue = PIECE_VALUES[pType] || 0;
                const hpRatio = hp / maxHp;
                // Score = material value * HP percentage
                return materialValue * hpRatio;
            };

            const renderEvalBar = (pieces, containerId, totalId) => {
                let total = 0;
                let pieceScores = [];

                // Group by piece type and sum scores
                const byType = {};
                pieces.forEach(p => {
                    const pType = getPieceType(p);
                    const score = calcPieceScore(p);
                    total += score;
                    if (!byType[pType]) byType[pType] = { count: 0, score: 0 };
                    byType[pType].count++;
                    byType[pType].score += score;
                });

                // Render individual piece scores (skip King since value=0)
                let piecesHtml = '';
                [1, 2, 3, 4, 5].forEach(pType => {  // Q, R, B, N, P
                    if (byType[pType]) {
                        const data = byType[pType];
                        piecesHtml += `<div class="piece-score">
                            <span class="piece-name">${PIECE_LETTERS[pType]}×${data.count}</span>
                            <span>${data.score.toFixed(1)}</span>
                        </div>`;
                    }
                });

                // Add King HP as separate indicator
                const king = pieces.find(p => getPieceType(p) === 0);
                if (king) {
                    const hp = getPieceHp(king);
                    const maxHp = getPieceMaxHp(king);
                    piecesHtml += `<div class="piece-score" style="margin-top: 8px; border-top: 1px solid; padding-top: 4px;">
                        <span class="piece-name">K HP</span>
                        <span>${hp}/${maxHp}</span>
                    </div>`;
                }

                document.getElementById(totalId).textContent = total.toFixed(1);
                document.getElementById(containerId).innerHTML = piecesHtml;
            };

            renderEvalBar(whitePieces, 'white-pieces', 'white-total');
            renderEvalBar(blackPieces, 'black-pieces', 'black-total');

            // Update advantage bar
            const whiteTotal = whitePieces.reduce((sum, p) => sum + calcPieceScore(p), 0);
            const blackTotal = blackPieces.reduce((sum, p) => sum + calcPieceScore(p), 0);
            const maxTotal = Math.max(whiteTotal + blackTotal, 1);
            const advantage = whiteTotal - blackTotal;

            const advantageFill = document.getElementById('advantage-fill');
            const advantageLabel = document.getElementById('advantage-label');

            if (Math.abs(advantage) < 0.5) {
                advantageFill.style.width = '0%';
                advantageFill.className = 'advantage-fill';
                advantageLabel.textContent = 'EVEN';
            } else if (advantage > 0) {
                // White ahead - fill extends right from center
                const pct = Math.min((advantage / maxTotal) * 100, 50);
                advantageFill.style.width = pct + '%';
                advantageFill.className = 'advantage-fill white-advantage';
                advantageLabel.textContent = `+${advantage.toFixed(1)} W`;
            } else {
                // Black ahead - fill extends left from center
                const pct = Math.min((Math.abs(advantage) / maxTotal) * 100, 50);
                advantageFill.style.width = pct + '%';
                advantageFill.className = 'advantage-fill black-advantage';
                advantageLabel.textContent = `+${Math.abs(advantage).toFixed(1)} B`;
            }
        }

        function reconstructState(upToMove) {
            // Start from initial state and apply moves
            const state = JSON.parse(JSON.stringify(currentGame.initial_state));

            for (let i = 0; i < upToMove; i++) {
                const move = currentGame.moves[i];
                applyMove(state, move, move.turn);
            }

            return state;
        }

        // Debug mode - set to true in console with: window.MOVE_DEBUG = true
        window.MOVE_DEBUG = false;
        const PIECE_CHARS = ['K','Q','R','B','N','P'];
        const TEAM_CHARS = ['W','B'];
        const MOVE_TYPE_NAMES = ['MOVE', 'ATTACK', 'MOVE_AND_ATTACK', 'ABILITY'];
        const ABILITY_NAMES = ['RoyalDecree', 'Overextend', 'Interpose', 'Consecration', 'Skirmish', 'Advance'];

        function pieceStr(p) {
            if (!p) return 'null';
            const t = TEAM_CHARS[normalizeTeam(p.team)] || '?';
            const pt = PIECE_CHARS[normalizePieceType(p.pieceType)] || '?';
            return `${t}${pt}@(${p.x},${p.y}) HP:${p.currentHp}/${p.maxHp || '?'}`;
        }

        function applyMove(state, move, turnNum) {
            // Move types: 0=MOVE, 1=ATTACK, 2=MOVE_AND_ATTACK (Knight), 3=ABILITY
            const MOVE = 0, ATTACK = 1, MOVE_AND_ATTACK = 2, ABILITY = 3;
            const debug = window.MOVE_DEBUG;

            // Find the piece - need to match by position AND verify piece type/team match
            // Move uses numeric format, pieces may use string format
            const expectedTeam = move.team;  // 0=White, 1=Black
            const expectedType = move.piece_type;  // 0=King, 1=Queen, etc.
            const moveDesc = `${TEAM_CHARS[expectedTeam]}${PIECE_CHARS[expectedType]} ${MOVE_TYPE_NAMES[move.move_type]}`;

            // First try strict matching: position + type + team
            let piece = state.pieces.find(p => {
                if (p.x !== move.from[0] || p.y !== move.from[1] || p.currentHp <= 0) return false;
                const pTeam = normalizeTeam(p.team);
                const pType = normalizePieceType(p.pieceType);
                return pTeam === expectedTeam && pType === expectedType;
            });

            // If strict match fails, try position-only match (for robustness)
            let usedFallback = false;
            if (!piece) {
                piece = state.pieces.find(p =>
                    p.x === move.from[0] && p.y === move.from[1] && p.currentHp > 0
                );
                if (piece) {
                    usedFallback = true;
                    console.warn(`Turn ${turnNum}: FALLBACK MATCH - expected ${moveDesc} at (${move.from[0]},${move.from[1]}), found ${pieceStr(piece)}`);
                }
            }

            if (!piece) {
                console.error(`Turn ${turnNum}: NO PIECE FOUND for ${moveDesc} at (${move.from[0]},${move.from[1]}) - STATE CORRUPTED`);
                // List all living pieces for debugging
                const living = state.pieces.filter(p => p.currentHp > 0);
                console.error(`  Living pieces (${living.length}): ${living.map(pieceStr).join(', ')}`);
                console.error(`  Move data:`, move);
                return;
            }

            if (debug) console.log(`Turn ${turnNum}: ${pieceStr(piece)} - ${MOVE_TYPE_NAMES[move.move_type]}${move.ability_id != null ? ' ' + ABILITY_NAMES[move.ability_id] : ''}`);

            const oldX = piece.x, oldY = piece.y;

            // Handle ability moves
            if (move.move_type === ABILITY && move.ability_id !== null && move.ability_id !== undefined) {
                // Ability IDs: 0=Royal Decree, 1=Overextend, 2=Interpose, 3=Consecration, 4=Skirmish, 5=Advance
                switch (move.ability_id) {
                    case 0:  // Royal Decree - just activates buff, no state change
                        break;
                    case 1:  // Overextend - move then attack, 2 self-damage
                        piece.x = move.to[0];
                        piece.y = move.to[1];
                        piece.currentHp -= 2;  // Self-damage from Overextend
                        if (debug) console.log(`  Overextend: moved to (${piece.x},${piece.y}), self-damage 2, HP now ${piece.currentHp}`);
                        // Attack damage applied below
                        break;
                    case 2:  // Interpose - no movement
                        break;
                    case 3:  // Consecration - heal ally
                        // Note: Healing amount not in move data, but ability_target has heal location
                        if (move.ability_target && move.ability_target[0] >= 0) {
                            const healTarget = state.pieces.find(p =>
                                p.x === move.ability_target[0] && p.y === move.ability_target[1] && p.currentHp > 0
                            );
                            if (healTarget && debug) console.log(`  Consecration target: ${pieceStr(healTarget)}`);
                        }
                        break;
                    case 4:  // Skirmish - attack then reposition
                        piece.x = move.to[0];
                        piece.y = move.to[1];
                        if (debug) console.log(`  Skirmish: repositioned to (${piece.x},${piece.y})`);
                        // Damage applied below
                        break;
                    case 5:  // Advance - move forward
                        piece.x = move.to[0];
                        piece.y = move.to[1];
                        if (debug) console.log(`  Advance: moved to (${piece.x},${piece.y})`);
                        break;
                }
            } else {
                // Only move the piece for MOVE and MOVE_AND_ATTACK types
                // ATTACK type means piece stays in place and attacks target
                if (move.move_type === MOVE || move.move_type === MOVE_AND_ATTACK) {
                    piece.x = move.to[0];
                    piece.y = move.to[1];
                    if (debug) console.log(`  Moved: (${oldX},${oldY}) -> (${piece.x},${piece.y})`);
                }
            }

            // Check for pawn promotion (pawn reaches last rank)
            const pieceTypeId = normalizePieceType(piece.pieceType);
            const teamId = normalizeTeam(piece.team);
            if (pieceTypeId === 5) {  // Pawn
                const promotionRank = teamId === 0 ? 7 : 0;
                if (piece.y === promotionRank) {
                    console.log(`Turn ${turnNum}: PROMOTION - ${pieceStr(piece)} promoted to Queen!`);
                    // Preserve string format if that's what was used
                    piece.pieceType = typeof piece.pieceType === 'string' ? 'Queen' : 1;
                    piece.wasPawn = true;
                }
            }

            // Apply damage to target
            if (move.damage > 0) {
                // Determine target position based on move type
                let targetPos = null;

                // Check if attack position is valid (not [-1,-1] sentinel)
                const hasValidAttack = move.attack &&
                    Array.isArray(move.attack) &&
                    move.attack[0] >= 0 && move.attack[1] >= 0;

                if (move.move_type === MOVE_AND_ATTACK && hasValidAttack) {
                    targetPos = move.attack;  // Knight combo
                } else if (move.move_type === ATTACK) {
                    targetPos = move.to;  // Regular attack - target is at 'to' position
                } else if (move.move_type === ABILITY && hasValidAttack) {
                    targetPos = move.attack;  // Ability with attack (Overextend, Skirmish)
                }

                if (targetPos) {
                    const target = state.pieces.find(p =>
                        p.x === targetPos[0] && p.y === targetPos[1] && p.currentHp > 0 && p !== piece
                    );
                    if (target) {
                        const oldHp = target.currentHp;
                        target.currentHp -= move.damage;
                        if (debug) console.log(`  Damage: ${pieceStr(target)} took ${move.damage} (${oldHp} -> ${target.currentHp})`);
                        if (target.currentHp <= 0) {
                            console.log(`Turn ${turnNum}: DEATH - ${TEAM_CHARS[normalizeTeam(target.team)]}${PIECE_CHARS[normalizePieceType(target.pieceType)]} at (${targetPos[0]},${targetPos[1]}) killed!`);
                        }
                    } else {
                        console.error(`Turn ${turnNum}: NO TARGET at (${targetPos[0]},${targetPos[1]}) for ${move.damage} damage - STATE CORRUPTED`);
                        const living = state.pieces.filter(p => p.currentHp > 0);
                        console.error(`  Living pieces: ${living.map(pieceStr).join(', ')}`);
                    }
                }
            }

            // Apply interpose damage to a defending Rook (if any)
            if (move.interpose_blocked && move.interpose_blocked > 0) {
                // Find which Rook on the defending team could have interposed
                // Interpose requires: same row or column as attack target, within 3 squares
                const defendingTeam = 1 - move.team;  // Opposite of attacker

                // Determine attack target position
                const hasValidAttack = move.attack && Array.isArray(move.attack) && move.attack[0] >= 0;
                const attackTargetPos = hasValidAttack ? move.attack :
                    (move.move_type === 1 ? move.to : null);  // ATTACK type targets 'to'

                if (attackTargetPos) {
                    const [tx, ty] = attackTargetPos;

                    // Check which Rook can interpose (orthogonal LOS within 3 squares)
                    const canInterpose = (rook) => {
                        const rx = rook.x, ry = rook.y;
                        if (rx === tx) return Math.abs(ry - ty) <= 3;  // Same column
                        if (ry === ty) return Math.abs(rx - tx) <= 3;  // Same row
                        return false;
                    };

                    const rooks = state.pieces.filter(p =>
                        p.currentHp > 0 &&
                        normalizeTeam(p.team) === defendingTeam &&
                        normalizePieceType(p.pieceType) === 2 &&  // Rook
                        canInterpose(p)
                    );

                    if (rooks.length > 0) {
                        // If multiple Rooks could interpose, pick one (game engine chose one)
                        const rook = rooks[0];
                        const oldHp = rook.currentHp;
                        rook.currentHp -= move.interpose_blocked;
                        console.log(`Turn ${turnNum}: INTERPOSE - ${pieceStr(rook)} blocked ${move.interpose_blocked} damage (${oldHp} -> ${rook.currentHp})`);
                        if (rook.currentHp <= 0) {
                            console.log(`Turn ${turnNum}: DEATH - Interposing Rook at (${rook.x},${rook.y}) killed!`);
                        }
                    } else {
                        console.warn(`Turn ${turnNum}: INTERPOSE ${move.interpose_blocked} but no Rook in LOS of (${tx},${ty})?`);
                    }
                }
            }
        }

        function firstMove() { currentMoveIndex = -1; renderPosition(); updateScrubber(); }
        function lastMove() { currentMoveIndex = currentGame.moves.length - 1; renderPosition(); updateScrubber(); }
        function prevMove() { if (currentMoveIndex > -1) { currentMoveIndex--; renderPosition(); updateScrubber(); } }
        function nextMove() { if (currentMoveIndex < currentGame.moves.length - 1) { currentMoveIndex++; renderPosition(); updateScrubber(); } }

        function scrubToMove(value) {
            currentMoveIndex = parseInt(value);
            renderPosition();
            updateScrubberLabels();
        }

        function updateScrubber() {
            const scrubber = document.getElementById('move-scrubber');
            if (scrubber && currentGame) {
                scrubber.min = -1;
                scrubber.max = currentGame.moves.length - 1;
                scrubber.value = currentMoveIndex;
                updateScrubberLabels();
            }
        }

        function updateScrubberLabels() {
            const posLabel = document.getElementById('scrubber-position');
            const totalLabel = document.getElementById('scrubber-total');
            if (posLabel && currentGame) {
                if (currentMoveIndex === -1) {
                    posLabel.textContent = 'Game Start';
                } else {
                    const actualTurn = Math.ceil((currentMoveIndex + 2) / 2);
                    posLabel.textContent = `Move ${currentMoveIndex + 1} (Turn ${actualTurn})`;
                }
            }
            if (totalLabel && currentGame) {
                const totalTurns = Math.ceil(currentGame.moves.length / 2);
                totalLabel.textContent = `${currentGame.moves.length} moves (${totalTurns} turns)`;
            }
        }

        function initScrubber() {
            const scrubber = document.getElementById('move-scrubber');
            if (scrubber && currentGame) {
                scrubber.min = -1;
                scrubber.max = currentGame.moves.length - 1;
                scrubber.value = -1;
                updateScrubberLabels();
            }
        }

        function toggleAutoplay() {
            if (autoplayInterval) {
                stopAutoplay();
            } else {
                document.getElementById('autoplay-btn').textContent = 'Pause';
                autoplayInterval = setInterval(() => {
                    if (currentMoveIndex < currentGame.moves.length - 1) {
                        currentMoveIndex++;
                        renderPosition();
                        updateScrubber();
                    } else {
                        stopAutoplay();
                    }
                }, autoplaySpeed);
            }
        }

        function stopAutoplay() {
            if (autoplayInterval) {
                clearInterval(autoplayInterval);
                autoplayInterval = null;
            }
            const btn = document.getElementById('autoplay-btn');
            if (btn) btn.textContent = 'Play';
        }

        function updateAutoplaySpeed(sliderValue) {
            autoplaySpeed = 2100 - parseInt(sliderValue);
            saveAutoplaySpeed(autoplaySpeed);

            // If currently autoplaying, restart with new speed
            if (autoplayInterval) {
                clearInterval(autoplayInterval);
                autoplayInterval = setInterval(() => {
                    if (currentMoveIndex < currentGame.moves.length - 1) {
                        currentMoveIndex++;
                        renderPosition();
                        updateScrubber();
                    } else {
                        stopAutoplay();
                    }
                }, autoplaySpeed);
            }
        }

        function startGameListRefresh() {
            // Stop any existing interval
            if (gameListRefreshInterval) {
                clearInterval(gameListRefreshInterval);
            }
            // Refresh game list periodically
            gameListRefreshInterval = setInterval(() => {
                loadGameList();
            }, REFRESH_INTERVAL_MS);
        }

        async function downloadAnalysis(lastN) {
            const statusEl = document.getElementById('analysis-status');
            statusEl.textContent = `Generating analysis for last ${lastN} iterations...`;
            statusEl.style.color = '#00d4ff';

            try {
                const response = await fetch(`/api/analysis?last=${lastN}`);
                if (!response.ok) {
                    throw new Error('Analysis failed');
                }

                // Get filename from Content-Disposition header
                const disposition = response.headers.get('Content-Disposition');
                let filename = `analysis_${lastN}.txt`;
                if (disposition) {
                    const match = disposition.match(/filename="?([^"]+)"?/);
                    if (match) filename = match[1];
                }

                // Download the file
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

                statusEl.textContent = `Downloaded: ${filename}`;
                statusEl.style.color = '#4caf50';
            } catch (error) {
                statusEl.textContent = `Error: ${error.message}`;
                statusEl.style.color = '#f44336';
            }
        }

        // Load games on start (reads URL params) and begin auto-refresh
        loadGameList();
        startGameListRefresh();
    </script>
</body>
</html>
"""


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def __init__(self, *args, replays_dir: str = "data/replays", **kwargs):
        self.replays_dir = replays_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_html(DASHBOARD_HTML)

        elif path == "/api/games":
            iteration = query.get("iteration", ["latest"])[0]
            outcome = query.get("outcome", ["all"])[0]
            self.send_json(self.get_game_list(iteration=iteration, outcome=outcome))

        elif path == "/api/game":
            game_path = query.get("path", [""])[0]
            if game_path and os.path.exists(game_path):
                with open(game_path) as f:
                    self.send_json(json.load(f))
            else:
                self.send_json({"error": "Game not found"})

        elif path == "/api/play":
            # Play a new game and return the replay
            replay = self.play_new_game()
            self.send_json(replay)

        elif path == "/api/analysis":
            # Run analysis and return as downloadable file
            last_n = int(query.get("last", ["50"])[0])
            # Validate last_n
            if last_n not in [50, 100, 200, 500, 1000]:
                last_n = 50
            self.run_and_send_analysis(last_n)

        else:
            self.send_error(404)

    def _load_iteration_metadata(self, iter_path: Path) -> dict:
        """Load or create metadata cache for an iteration folder.

        Returns dict: {game_num: {"winner": str, "moves": int}, ...}
        """
        cache_path = iter_path / ".metadata.json"
        cache = {}

        # Load existing cache
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cache = json.load(f)
            except:
                cache = {}

        # Find all game files in this iteration
        game_files = list(iter_path.glob("game_*.json"))
        cache_updated = False

        for gf in game_files:
            game_part = gf.stem.replace("game_", "")
            try:
                game_num = int(game_part)
            except ValueError:
                continue

            # Skip if already cached
            if str(game_num) in cache:
                continue

            # Read game file to extract metadata (only for new games)
            try:
                with open(gf) as f:
                    data = json.load(f)
                winner_val = data.get("winner")
                # Handle both int (0/1) and string ("White"/"Black") formats
                if winner_val in (0, "White", "WHITE"):
                    winner = "WHITE"
                elif winner_val in (1, "Black", "BLACK"):
                    winner = "BLACK"
                else:
                    winner = "DRAW"
                moves = len(data.get("moves", []))
                cache[str(game_num)] = {"winner": winner, "moves": moves}
                cache_updated = True
            except:
                pass

        # Save updated cache
        if cache_updated:
            try:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
            except:
                pass

        return cache

    def get_game_list(self, iteration: str = "latest", outcome: str = "all") -> dict:
        """Get filtered list of saved replays with metadata.

        Uses per-iteration metadata cache to avoid reading all game files.
        Only reads game content for files not yet in cache.

        Args:
            iteration: "latest", "all", or iteration number as string
            outcome: "all", "decisive", "white", "black", or "draw"

        Returns:
            Dict with games, available_iterations, and current_iteration
        """
        all_games = []
        iterations_set = set()
        replays_path = Path(self.replays_dir)

        if replays_path.exists():
            # Find all iteration folders
            iter_folders = [d for d in replays_path.iterdir() if d.is_dir() and d.name.startswith("iter_")]

            for iter_folder in iter_folders:
                try:
                    iter_num = int(iter_folder.name.replace("iter_", ""))
                    iterations_set.add(iter_num)
                except ValueError:
                    continue

                # Load metadata from cache (fast)
                metadata = self._load_iteration_metadata(iter_folder)

                for game_num_str, meta in metadata.items():
                    game_num = int(game_num_str)
                    game_path = iter_folder / f"game_{game_num:04d}.json"

                    all_games.append({
                        "id": f"{iter_num}_{game_num}",
                        "path": str(game_path),
                        "moves": meta["moves"],
                        "winner": meta["winner"],
                        "iteration": iter_num,
                        "game_num": game_num,
                    })

            # Also handle legacy games not in iter_* folders
            legacy_files = [f for f in replays_path.glob("*.json") if f.is_file()]
            for f in legacy_files:
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                    winner_val = data.get("winner")
                    if winner_val in (0, "White", "WHITE"):
                        winner = "WHITE"
                    elif winner_val in (1, "Black", "BLACK"):
                        winner = "BLACK"
                    else:
                        winner = "DRAW"
                    all_games.append({
                        "id": f"legacy_{f.stem}",
                        "path": str(f),
                        "moves": len(data.get("moves", [])),
                        "winner": winner,
                        "iteration": -1,
                        "game_num": 0,
                    })
                except:
                    pass

        # Determine available iterations (sorted descending)
        available_iterations = sorted(iterations_set, reverse=True)

        # Resolve "latest" to actual iteration number
        if iteration == "latest":
            current_iteration = available_iterations[0] if available_iterations else -1
        elif iteration == "all":
            current_iteration = "all"
        else:
            current_iteration = int(iteration)

        # Filter by iteration
        if current_iteration == "all":
            filtered_games = all_games
        else:
            filtered_games = [g for g in all_games if g["iteration"] == current_iteration]

        # Filter by outcome
        if outcome == "decisive":
            filtered_games = [g for g in filtered_games if g["winner"] in ("WHITE", "BLACK")]
        elif outcome == "white":
            filtered_games = [g for g in filtered_games if g["winner"] == "WHITE"]
        elif outcome == "black":
            filtered_games = [g for g in filtered_games if g["winner"] == "BLACK"]
        elif outcome == "draw":
            filtered_games = [g for g in filtered_games if g["winner"] == "DRAW"]

        # Sort by game number within iteration
        filtered_games.sort(key=lambda g: g["game_num"])

        return {
            "games": filtered_games,
            "available_iterations": available_iterations,
            "current_iteration": current_iteration,
            "total_count": len(filtered_games),
        }

    def play_new_game(self) -> dict:
        """Play a new random game and save it."""
        sim = GameSimulator()
        sim.set_seed(int(time.time() * 1000) % 1000000)

        from .game_viewer import play_and_record
        replay = play_and_record(simulator=sim, verbose=False)

        # Save replay
        os.makedirs(self.replays_dir, exist_ok=True)
        replay_path = f"{self.replays_dir}/game_{int(time.time())}.json"
        replay.save(replay_path)

        # Return as dict
        from dataclasses import asdict
        return asdict(replay)

    def send_html(self, content: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(content.encode())

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_file(self, content: str, filename: str, content_type: str = "text/plain"):
        """Send content as a downloadable file."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(content.encode())))
        self.end_headers()
        self.wfile.write(content.encode())

    def run_and_send_analysis(self, last_n: int):
        """Run the analysis script and return results as downloadable file."""
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_analysis_last{last_n}.txt"

        # Create analysis directory if it doesn't exist
        analysis_dir = Path(self.replays_dir).parent / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        output_path = analysis_dir / filename

        try:
            # Run the analysis script and capture output
            # We need to import and run it directly to capture stdout
            from io import StringIO
            import contextlib

            # Capture stdout
            output_buffer = StringIO()

            # Import the analysis module
            from scripts import analyze_replays

            # Temporarily replace sys.argv
            old_argv = sys.argv
            sys.argv = ['analyze_replays', '--replays', self.replays_dir, '--last', str(last_n)]

            # Capture stdout
            with contextlib.redirect_stdout(output_buffer):
                try:
                    analyze_replays.main()
                except SystemExit:
                    pass  # Ignore sys.exit calls

            sys.argv = old_argv

            # Get the output
            output = output_buffer.getvalue()

            # Save to file
            with open(output_path, 'w') as f:
                f.write(output)

            # Send the file
            self.send_file(output, filename)

        except Exception as e:
            # If the import approach fails, try subprocess
            try:
                # Find the scripts directory
                scripts_dir = Path(__file__).parent.parent / "scripts"
                analyze_script = scripts_dir / "analyze_replays.py"

                if analyze_script.exists():
                    result = subprocess.run(
                        [sys.executable, str(analyze_script),
                         '--replays', self.replays_dir,
                         '--last', str(last_n)],
                        capture_output=True,
                        text=True,
                        timeout=120  # 2 minute timeout
                    )
                    output = result.stdout

                    # Save to file
                    with open(output_path, 'w') as f:
                        f.write(output)

                    self.send_file(output, filename)
                else:
                    self.send_response(500)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(f"Analysis script not found: {analyze_script}".encode())
            except Exception as e2:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Analysis failed: {str(e)} / {str(e2)}".encode())

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def create_handler(replays_dir: str):
    """Create handler class with custom replays directory."""
    class Handler(DashboardHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, replays_dir=replays_dir, **kwargs)
    return Handler


def get_local_ip() -> str:
    """Get the local IP address for network access."""
    import socket
    try:
        # Connect to a public DNS to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def run_dashboard(port: int = 5000, replays_dir: str = "data/replays", host: str = "0.0.0.0"):
    """Start the dashboard server."""
    handler = create_handler(replays_dir)
    server = HTTPServer((host, port), handler)

    local_ip = get_local_ip()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           EXCHANGE Training Dashboard                     ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║   Local:    http://localhost:{port:<5}                      ║
║   Network:  http://{local_ip}:{port:<5}                    ║
║                                                          ║
║   📱 Access from phone using the Network URL above!      ║
║                                                          ║
║   Features:                                              ║
║   • Browse saved game replays                            ║
║   • Step through moves with Next/Prev                    ║
║   • Watch new games play out                             ║
║   • See HP bars and damage                               ║
║                                                          ║
║   Press Ctrl+C to stop                                   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EXCHANGE Training Dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on (default: 5000)")
    parser.add_argument("--replays", type=str, default="runs/experiment/replays", help="Replays directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0 for network access)")
    args = parser.parse_args()

    run_dashboard(port=args.port, replays_dir=args.replays, host=args.host)


if __name__ == "__main__":
    main()
