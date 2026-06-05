"""HanoiCrossing game engine — work in progress."""

from __future__ import annotations

from .models import GameState, Move, Player, VISIBLE


class HanoiCrossing:
    """
    Game engine for Hanoi Crossing.

    A gets odd disks (1, 3, 5, ...) on pole 1a; goal: move all to 3a.
    B gets even disks (2, 4, 6, ...) on pole 1b; goal: move all to 3b.
    Pole 2 is shared — both players can see and use it.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n

    def initial_state(self) -> GameState:
        """Starting position: A's odd disks on 1a, B's even disks on 1b."""
        a_disks = list(range(2 * self.n - 1, 0, -2))
        b_disks = list(range(2 * self.n, 0, -2))
        return GameState(
            poles={"1a": a_disks, "2": [], "3a": [], "1b": b_disks, "3b": []},
            hands={Player.A: None, Player.B: None},
        )

    def legal_moves(self, state: GameState, player: Player) -> list[Move]:
        """All moves the player can legally make. Skip is always included."""
        moves: list[Move] = [Move.skip()]
        hand = state.hands[player]
        visible = VISIBLE[player]

        if hand is None:
            for pole in visible:
                if state.poles[pole]:
                    moves.append(Move.lift(pole))
        else:
            for pole in visible:
                stack = state.poles[pole]
                if not stack or stack[-1] > hand:
                    moves.append(Move.place(pole))

        return moves
