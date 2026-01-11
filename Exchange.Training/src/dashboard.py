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
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import threading
import time

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

        // Watched games tracking
        function getWatchedGames() {
            const stored = localStorage.getItem('watchedGames');
            return stored ? JSON.parse(stored) : [];
        }

        function markGameAsWatched(gamePath) {
            const watched = getWatchedGames();
            if (!watched.includes(gamePath)) {
                watched.push(gamePath);
                localStorage.setItem('watchedGames', JSON.stringify(watched));
            }
        }

        function isGameWatched(gamePath) {
            return getWatchedGames().includes(gamePath);
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
                const watched = isGameWatched(g.path);
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
                    <li class="${watchClass} ${isActive}" onclick="loadGame('${g.path}')">
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

        async function loadGame(path) {
            stopAutoplay();
            const response = await fetch('/api/game?path=' + encodeURIComponent(path));
            currentGame = await response.json();
            currentGamePath = path;
            currentMoveIndex = -1;

            // Mark as watched and refresh list to update status
            markGameAsWatched(path);
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
                    <div class="board-wrapper">
                        <div class="board" id="board"></div>
                        <svg class="move-overlay" id="move-overlay" width="409" height="409"></svg>
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

            // Render board
            let html = '';
            for (let y = 7; y >= 0; y--) {
                for (let x = 0; x < 8; x++) {
                    const isLight = (x + y) % 2 === 1;
                    const piece = state.pieces.find(p => p.x === x && p.y === y && p.currentHp > 0);

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
                        let symbol = PIECE_LETTERS[piece.pieceType];
                        if (piece.wasPawn && piece.pieceType !== 5) {
                            symbol = 'P' + symbol;  // e.g., "PQ" for promoted Queen
                        }
                        const colorClass = piece.team === 0 ? 'white' : 'black';
                        const hpPct = (piece.currentHp / piece.maxHp) * 100;
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
                    let attackTarget;
                    if (move.attack) {
                        attackTarget = getCenter(move.attack[0], move.attack[1]);
                    } else if (move.move_type === 1) {
                        // Regular attack - 'to' is the target
                        attackTarget = getCenter(move.to[0], move.to[1]);
                    }
                    if (attackTarget) {
                        // Draw from piece's current position (after move)
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
            const whitePieces = state.pieces.filter(p => p.team === 0 && p.currentHp > 0);
            const blackPieces = state.pieces.filter(p => p.team === 1 && p.currentHp > 0);
            const pieceNames = ['K', 'Q', 'R', 'B', 'N', 'P'];

            // Helper to get piece symbol (with P prefix for promoted)
            const getPieceSymbol = (p) => {
                let sym = pieceNames[p.pieceType];
                if (p.wasPawn && p.pieceType !== 5) sym = 'P' + sym;
                return sym;
            };

            pieceList.innerHTML = `
                <span class="white">WHITE: ${whitePieces.map(p => getPieceSymbol(p) + '(' + p.currentHp + ')').join(' ')}</span>
                <span class="black">BLACK: ${blackPieces.map(p => getPieceSymbol(p) + '(' + p.currentHp + ')').join(' ')}</span>
            `;
        }

        function reconstructState(upToMove) {
            // Start from initial state and apply moves
            const state = JSON.parse(JSON.stringify(currentGame.initial_state));

            for (let i = 0; i < upToMove; i++) {
                const move = currentGame.moves[i];
                applyMove(state, move);
            }

            return state;
        }

        function applyMove(state, move) {
            // Move types: 0=MOVE, 1=ATTACK, 2=MOVE_AND_ATTACK (Knight), 3=ABILITY
            const MOVE = 0, ATTACK = 1, MOVE_AND_ATTACK = 2, ABILITY = 3;

            const piece = state.pieces.find(p =>
                p.x === move.from[0] && p.y === move.from[1] && p.currentHp > 0
            );
            if (!piece) return;

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
                        // Attack damage applied below
                        break;
                    case 2:  // Interpose - no movement
                        break;
                    case 3:  // Consecration - heal ally
                        // Healing not tracked in move data, skip
                        break;
                    case 4:  // Skirmish - attack then reposition
                        piece.x = move.to[0];
                        piece.y = move.to[1];
                        // Damage applied below
                        break;
                    case 5:  // Advance - move forward
                        piece.x = move.to[0];
                        piece.y = move.to[1];
                        break;
                }
            } else {
                // Only move the piece for MOVE and MOVE_AND_ATTACK types
                // ATTACK type means piece stays in place and attacks target
                if (move.move_type === MOVE || move.move_type === MOVE_AND_ATTACK) {
                    piece.x = move.to[0];
                    piece.y = move.to[1];
                }
            }

            // Check for pawn promotion (pawn reaches last rank)
            if (piece.pieceType === 5) {  // Pawn
                const promotionRank = piece.team === 0 ? 7 : 0;
                if (piece.y === promotionRank) {
                    piece.pieceType = 1;  // Promote to Queen
                    piece.wasPawn = true;
                }
            }

            // Apply damage
            if (move.damage > 0) {
                // Determine target position based on move type
                let targetPos = null;
                if (move.move_type === MOVE_AND_ATTACK || (move.move_type === ABILITY && move.attack)) {
                    targetPos = move.attack;  // Knight combo or Skirmish
                } else if (move.move_type === ATTACK) {
                    targetPos = move.to;  // Regular attack
                } else if (move.move_type === ABILITY && move.ability_id === 1 && move.attack) {
                    targetPos = move.attack;  // Overextend
                }
                if (targetPos) {
                    const target = state.pieces.find(p =>
                        p.x === targetPos[0] && p.y === targetPos[1] && p.currentHp > 0 && p !== piece
                    );
                    if (target) {
                        target.currentHp -= move.damage;
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

        else:
            self.send_error(404)

    def get_game_list(self, iteration: str = "latest", outcome: str = "all") -> dict:
        """Get filtered list of saved replays with metadata.

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
            # Find all JSON files recursively (handles iter_XXXX/game_XXXX.json structure)
            all_files = list(replays_path.glob("**/*.json"))
            # Sort by modification time (newest first)
            all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            for f in all_files:
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                    winner = "WHITE" if data.get("winner") == 0 else "BLACK" if data.get("winner") == 1 else "DRAW"

                    # Extract iteration and game number from path
                    rel_path = f.relative_to(replays_path)
                    parts = rel_path.parts
                    if len(parts) >= 2:
                        iter_part = parts[-2].replace("iter_", "")
                        game_part = parts[-1].replace("game_", "").replace(".json", "")
                        game_id = f"{winner}_{iter_part}_{game_part}"
                        iter_num = int(iter_part)
                        game_num = int(game_part)
                    else:
                        game_id = f"{winner}_{f.stem}"
                        iter_num = -1
                        game_num = 0

                    if iter_num >= 0:
                        iterations_set.add(iter_num)

                    all_games.append({
                        "id": game_id,
                        "path": str(f),
                        "moves": len(data.get("moves", [])),
                        "winner": winner,
                        "iteration": iter_num,
                        "game_num": game_num,
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
