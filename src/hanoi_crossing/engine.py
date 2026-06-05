"""
HanoiCrossing game engine.

Public API:
  initial_state()            -> starting GameState
  legal_moves(state, player) -> all valid moves from this position
  apply(state, player, move) -> new state; original is never mutated
  is_over(state)             -> bool
"""

from __future__ import annotations

from typing import Optional

from .models import Action, GameState, Move, Player, GOAL_POLE, VISIBLE


class HanoiCrossing:
    """
    Game engine for Hanoi Crossing.

    A gets odd disks (1, 3, 5, ...) on pole 1a, must move them all to 3a.
    B gets even disks (2, 4, 6, ...) on pole 1b, must move them all to 3b.
    Pole 2 is shared — both players can see and use it.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n

    # -- Public API ---------------------------------------------------------------

    def initial_state(self) -> GameState:
        """Starting position: A's odd disks on 1a, B's even disks on 1b."""
        a_disks = list(range(2 * self.n - 1, 0, -2))   # [2n-1, ..., 3, 1]
        b_disks = list(range(2 * self.n, 0, -2))         # [2n,   ..., 4, 2]
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

    def apply(self, state: GameState, player: Player, move: Move) -> GameState:
        """Apply move and return the new state. Illegal moves waste the turn but don't change the board."""
        new = state.copy()
        new.turn += 1

        if not self._is_legal(state, player, move):
            return new

        match move.action:
            case Action.LIFT:
                new.hands[player] = new.poles[move.pole].pop()
            case Action.PLACE:
                new.poles[move.pole].append(new.hands[player])
                new.hands[player] = None
            case Action.SKIP:
                pass

        new.winner = self._find_winner(new)
        return new

    def is_over(self, state: GameState) -> bool:
        return state.winner is not None

    # -- Private helpers ----------------------------------------------------------

    def _is_legal(self, state: GameState, player: Player, move: Move) -> bool:
        """Direct O(1) validation — avoids generating all legal moves just to check one."""
        match move.action:
            case Action.SKIP:
                return True
            case Action.LIFT:
                return (
                    move.pole in VISIBLE[player]
                    and state.hands[player] is None
                    and bool(state.poles[move.pole])
                )
            case Action.PLACE:
                if move.pole not in VISIBLE[player]:
                    return False
                disk = state.hands[player]
                if disk is None:
                    return False
                stack = state.poles[move.pole]
                return not stack or stack[-1] > disk
        return False

    def _find_winner(self, state: GameState) -> Optional[str]:
        """Returns the first player whose win condition is satisfied, or None."""
        for player in Player:
            if self._has_won(state, player):
                return player
        return None

    def _has_won(self, state: GameState, player: Player) -> bool:
        """Win = hand empty, goal pole non-empty, all other visible poles empty."""
        if state.hands[player] is not None:
            return False
        goal = GOAL_POLE[player]
        if not state.poles[goal]:
            return False
        return all(not state.poles[p] for p in VISIBLE[player] if p != goal)
