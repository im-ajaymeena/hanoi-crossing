"""
Random-play CLI — both players make random valid moves until someone wins
or the turn limit is reached.

The turn order itself is also randomised by default (each step independently
picks A or B with equal probability), satisfying the requirement that
the engine must not assume any fixed pattern.

Usage:
    hanoi-random              # n=2, up to 500 turns, random seed
    hanoi-random 3            # n=3
    hanoi-random 3 1000 42    # n=3, max 1000 turns, seed=42 (reproducible)
"""

from __future__ import annotations

import random
import sys

from ..engine import HanoiCrossing
from ..models import Player


def main() -> None:
    n         = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    seed      = int(sys.argv[3]) if len(sys.argv) > 3 else None

    rng = random.Random(seed)
    engine = HanoiCrossing(n)
    state = engine.initial_state()

    print(f"=== Hanoi Crossing Random Play  (n={n}, seed={seed}) ===\n")

    for turn_num in range(1, max_turns + 1):
        player = rng.choice([Player.A, Player.B])
        move   = rng.choice(sorted(engine.legal_moves(state, player), key=str))

        state = engine.apply(state, player, move)
        print(f"Turn {turn_num:4d} · Player {player} → {move}")

        if engine.is_over(state):
            print(f"\nResult: Player {state.winner} wins on turn {turn_num}!")
            return

    print(f"\nResult: No winner after {max_turns} turns.")
