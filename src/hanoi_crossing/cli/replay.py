"""
Replay CLI — reads a YAML file of pre-recorded moves and plays them through
the engine, printing each step and the final state.

Input format (YAML):
    n: 1
    turns:
      - [A, lift, 1a]   # [player, action, pole]
      - [B, lift, 1b]
      - [A, place, 3a]
      - [A, skip]       # pole is omitted for skip

Usage:
    hanoi-replay examples/n1_example.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml

from ..engine import HanoiCrossing
from ..models import Action, GameState, Move, Player


# ── Formatting ────────────────────────────────────────────────────────────────


def _render_stack(stack: list[int]) -> str:
    return f"[{', '.join(str(d) for d in stack)}]" if stack else "[]"


def _render_state(state: GameState) -> str:
    poles = state.poles
    hand_a = state.hands[Player.A]
    hand_b = state.hands[Player.B]
    return (
        f"  Poles │ 1a:{_render_stack(poles['1a'])}  "
        f"2:{_render_stack(poles['2'])}  "
        f"3a:{_render_stack(poles['3a'])}  │  "
        f"1b:{_render_stack(poles['1b'])}  "
        f"3b:{_render_stack(poles['3b'])}\n"
        f"  Hands │ A={hand_a}  B={hand_b}"
    )


# ── Parsing ───────────────────────────────────────────────────────────────────


def _parse_turn(raw: list) -> tuple[Player, Move]:
    """Parse [player, action] or [player, action, pole] into (Player, Move)."""
    player = Player(raw[0])
    action = Action(raw[1])
    pole: Optional[str] = raw[2] if len(raw) > 2 else None
    return player, Move(action, pole)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: hanoi-replay <replay_file.yaml>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    data = yaml.safe_load(path.read_text())
    n: int = data["n"]
    raw_turns: list = data["turns"]

    engine = HanoiCrossing(n)
    state = engine.initial_state()

    print(f"=== Hanoi Crossing Replay  (n={n}) ===\n")
    print(f"Initial state (turn 0):")
    print(_render_state(state))

    for i, raw in enumerate(raw_turns, start=1):
        player, move = _parse_turn(raw)
        prev_state = state
        state = engine.apply(state, player, move)

        legal = move in engine.legal_moves(prev_state, player)
        tag = "" if legal else "  [ILLEGAL — turn wasted]"
        print(f"\nTurn {i} · Player {player} → {move}{tag}")
        print(_render_state(state))

        if engine.is_over(state):
            print(f"\nResult: Player {state.winner} wins on turn {i}!")
            return

    print(f"\nResult: No winner after {len(raw_turns)} turns.")
