"""
Data types for Hanoi Crossing.

Board layout (pole 2 is shared):
        1a
        |
 1b -- [2] -- 3b
        |
        3a

Player A sees: 1a, 2, 3a  (odd disks, goal = 3a)
Player B sees: 1b, 2, 3b  (even disks, goal = 3b)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class Player(StrEnum):
    A = "A"
    B = "B"


class Action(StrEnum):
    LIFT  = "lift"
    PLACE = "place"
    SKIP  = "skip"


# Poles each player can see and interact with
VISIBLE: dict[Player, frozenset[str]] = {
    Player.A: frozenset({"1a", "2", "3a"}),
    Player.B: frozenset({"1b", "2", "3b"}),
}

# Destination pole each player must fill to win
GOAL_POLE: dict[Player, str] = {
    Player.A: "3a",
    Player.B: "3b",
}


@dataclass(frozen=True)
class Move:
    """A single player action — lift, place, or skip."""

    action: Action
    pole: Optional[str] = None  # None for skip

    @classmethod
    def lift(cls, pole: str) -> Move:
        return cls(Action.LIFT, pole)

    @classmethod
    def place(cls, pole: str) -> Move:
        return cls(Action.PLACE, pole)

    @classmethod
    def skip(cls) -> Move:
        return cls(Action.SKIP)

    def __str__(self) -> str:
        return "skip" if self.action == Action.SKIP else f"{self.action}({self.pole})"


@dataclass
class GameState:
    """
    Snapshot of the board. Treat as immutable — use .copy() before mutating.

    Pole stacks are stored bottom-to-top (index 0 = largest disk).
    """

    poles: dict[str, list[int]]       # pole_id -> disk stack
    hands: dict[str, Optional[int]]   # player -> held disk size, or None
    winner: Optional[str] = None
    turn: int = 0

    def copy(self) -> GameState:
        return GameState(
            poles={pole: list(stack) for pole, stack in self.poles.items()},
            hands=dict(self.hands),
            winner=self.winner,
            turn=self.turn,
        )
