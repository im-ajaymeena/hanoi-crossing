"""
Tests for the HanoiCrossing engine.

All tests call the engine directly — no CLI or I/O involved.
Organised by concern: initial state → legal moves → apply → win conditions → shared pole.
"""

import pytest

from hanoi_crossing import HanoiCrossing, Move, Player
from hanoi_crossing.models import Action


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def e1() -> HanoiCrossing:
    return HanoiCrossing(n=1)


@pytest.fixture
def e2() -> HanoiCrossing:
    return HanoiCrossing(n=2)


# ── Initial state ─────────────────────────────────────────────────────────────


class TestInitialState:
    def test_n1_pole_contents(self, e1):
        s = e1.initial_state()
        assert s.poles["1a"] == [1]
        assert s.poles["1b"] == [2]
        assert s.poles["2"]  == []
        assert s.poles["3a"] == []
        assert s.poles["3b"] == []

    def test_n2_pole_contents(self, e2):
        s = e2.initial_state()
        # Largest disk at bottom (index 0), smallest at top
        assert s.poles["1a"] == [3, 1]
        assert s.poles["1b"] == [4, 2]

    def test_n3_pole_contents(self):
        s = HanoiCrossing(n=3).initial_state()
        assert s.poles["1a"] == [5, 3, 1]
        assert s.poles["1b"] == [6, 4, 2]

    def test_hands_empty_at_start(self, e1):
        s = e1.initial_state()
        assert s.hands[Player.A] is None
        assert s.hands[Player.B] is None

    def test_no_winner_at_start(self, e1):
        assert e1.initial_state().winner is None

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            HanoiCrossing(n=0)


# ── Legal moves ───────────────────────────────────────────────────────────────


class TestLegalMoves:
    def test_skip_is_always_legal(self, e1):
        s = e1.initial_state()
        assert Move.skip() in e1.legal_moves(s, Player.A)
        assert Move.skip() in e1.legal_moves(s, Player.B)

    def test_can_lift_from_own_start_pole(self, e1):
        s = e1.initial_state()
        assert Move.lift("1a") in e1.legal_moves(s, Player.A)
        assert Move.lift("1b") in e1.legal_moves(s, Player.B)

    def test_cannot_lift_from_opponents_private_pole(self, e1):
        s = e1.initial_state()
        moves_a = e1.legal_moves(s, Player.A)
        assert Move.lift("1b") not in moves_a
        assert Move.lift("3b") not in moves_a

    def test_cannot_lift_from_empty_pole(self, e1):
        s = e1.initial_state()
        # 3a starts empty
        assert Move.lift("3a") not in e1.legal_moves(s, Player.A)

    def test_cannot_lift_when_already_holding(self, e1):
        s = e1.initial_state()
        s = e1.apply(s, Player.A, Move.lift("1a"))
        moves = e1.legal_moves(s, Player.A)
        assert not any(m.action == Action.LIFT for m in moves)

    def test_can_only_place_when_holding(self, e1):
        s = e1.initial_state()
        # Hand is empty — no place moves
        moves = e1.legal_moves(s, Player.A)
        assert not any(m.action == Action.PLACE for m in moves)

    def test_cannot_place_larger_on_smaller(self, e2):
        s = e2.initial_state()
        s = e2.apply(s, Player.A, Move.lift("1a"))   # picks up disk 1
        s = e2.apply(s, Player.A, Move.place("3a"))  # puts 1 on 3a
        s = e2.apply(s, Player.A, Move.lift("1a"))   # picks up disk 3
        # 3a has [1] on top; 3 > 1 is False → can't place
        assert Move.place("3a") not in e2.legal_moves(s, Player.A)

    def test_can_place_smaller_on_larger(self, e2):
        s = e2.initial_state()
        s = e2.apply(s, Player.A, Move.lift("1a"))   # picks up disk 1 (smallest)
        # 1a now has [3] on top; 3 > 1 → can place back
        assert Move.place("1a") in e2.legal_moves(s, Player.A)

    def test_can_place_on_empty_pole(self, e1):
        s = e1.initial_state()
        s = e1.apply(s, Player.A, Move.lift("1a"))
        assert Move.place("3a") in e1.legal_moves(s, Player.A)
        assert Move.place("2")  in e1.legal_moves(s, Player.A)


# ── Apply ─────────────────────────────────────────────────────────────────────


class TestApply:
    def test_lift_moves_disk_to_hand(self, e1):
        s = e1.initial_state()
        s = e1.apply(s, Player.A, Move.lift("1a"))
        assert s.hands[Player.A] == 1
        assert s.poles["1a"] == []

    def test_place_moves_disk_from_hand_to_pole(self, e1):
        s = e1.initial_state()
        s = e1.apply(s, Player.A, Move.lift("1a"))
        s = e1.apply(s, Player.A, Move.place("3a"))
        assert s.hands[Player.A] is None
        assert s.poles["3a"] == [1]

    def test_skip_changes_nothing_on_board(self, e1):
        s = e1.initial_state()
        s2 = e1.apply(s, Player.A, Move.skip())
        assert s2.poles == s.poles
        assert s2.hands == s.hands

    def test_illegal_move_leaves_board_unchanged(self, e1):
        s = e1.initial_state()
        # Place without holding anything is illegal
        s2 = e1.apply(s, Player.A, Move.place("3a"))
        assert s2.poles == s.poles
        assert s2.hands == s.hands

    def test_illegal_move_still_increments_turn(self, e1):
        s = e1.initial_state()
        s2 = e1.apply(s, Player.A, Move.place("3a"))
        assert s2.turn == s.turn + 1

    def test_apply_does_not_mutate_original_state(self, e1):
        s = e1.initial_state()
        poles_before = {k: list(v) for k, v in s.poles.items()}
        e1.apply(s, Player.A, Move.lift("1a"))
        assert s.poles == poles_before  # original untouched

    def test_turn_counter_increments_on_every_call(self, e1):
        s = e1.initial_state()
        assert s.turn == 0
        s = e1.apply(s, Player.A, Move.skip())
        assert s.turn == 1
        s = e1.apply(s, Player.B, Move.skip())
        assert s.turn == 2


# ── Win conditions ────────────────────────────────────────────────────────────


class TestWinCondition:
    def test_example_n1_player_a_wins(self, e1):
        """The canonical N=1 walkthrough: A wins on turn 3."""
        s = e1.initial_state()
        s = e1.apply(s, Player.A, Move.lift("1a"))    # A lifts disk 1
        s = e1.apply(s, Player.B, Move.lift("1b"))    # B lifts disk 2
        s = e1.apply(s, Player.A, Move.place("3a"))   # A places disk 1 → wins
        assert s.winner == Player.A

    def test_b_can_win(self, e1):
        s = e1.initial_state()
        s = e1.apply(s, Player.B, Move.lift("1b"))
        s = e1.apply(s, Player.B, Move.place("3b"))
        assert s.winner == Player.B

    def test_no_win_while_holding_disk(self, e1):
        s = e1.initial_state()
        s = e1.apply(s, Player.A, Move.lift("1a"))
        assert s.winner is None

    def test_no_win_until_goal_pole_is_occupied(self, e1):
        s = e1.initial_state()
        # A skips — hand empty, 1a still has disk, not a win
        s = e1.apply(s, Player.A, Move.skip())
        assert s.winner is None

    def test_shared_pole_must_be_clear_for_a_to_win(self, e1):
        """B parks a disk on pole 2 — A cannot win until it's removed."""
        s = e1.initial_state()
        s = e1.apply(s, Player.B, Move.lift("1b"))   # B lifts disk 2
        s = e1.apply(s, Player.B, Move.place("2"))   # B parks disk 2 on shared pole
        s = e1.apply(s, Player.A, Move.lift("1a"))   # A lifts disk 1
        s = e1.apply(s, Player.A, Move.place("3a"))  # A places disk 1 on goal
        # A's visible poles: 1a=[] 2=[2] 3a=[1] — pole 2 not clear → no win
        assert s.winner is None

    def test_a_wins_the_moment_shared_pole_is_cleared(self, e1):
        """A's win is triggered by B's move — the engine checks both players after every turn."""
        s = e1.initial_state()
        s = e1.apply(s, Player.B, Move.lift("1b"))   # B lifts 2
        s = e1.apply(s, Player.B, Move.place("2"))   # B parks disk 2 on pole 2
        s = e1.apply(s, Player.A, Move.lift("1a"))   # A lifts 1
        s = e1.apply(s, Player.A, Move.place("3a"))  # A places 1 on 3a — no win: pole 2 still blocked
        assert s.winner is None
        s = e1.apply(s, Player.B, Move.lift("2"))    # B lifts disk 2 off pole 2 → pole 2 now empty
        # Now: A hand=None, 1a=[], 2=[], 3a=[1] — A's win condition is satisfied on B's turn
        assert s.winner == Player.A

    def test_is_over_reflects_winner(self, e1):
        s = e1.initial_state()
        assert not e1.is_over(s)
        s = e1.apply(s, Player.A, Move.lift("1a"))
        s = e1.apply(s, Player.A, Move.place("3a"))
        assert e1.is_over(s)


# ── Shared pole interactions ──────────────────────────────────────────────────


class TestSharedPole:
    def test_a_can_lift_disk_b_placed_on_pole2(self, e1):
        s = e1.initial_state()
        s = e1.apply(s, Player.B, Move.lift("1b"))
        s = e1.apply(s, Player.B, Move.place("2"))
        # A can now see and lift disk 2 from pole 2
        assert Move.lift("2") in e1.legal_moves(s, Player.A)

    def test_b_can_lift_disk_a_placed_on_pole2(self, e1):
        s = e1.initial_state()
        s = e1.apply(s, Player.A, Move.lift("1a"))
        s = e1.apply(s, Player.A, Move.place("2"))
        assert Move.lift("2") in e1.legal_moves(s, Player.B)

    def test_hanoi_rule_enforced_on_shared_pole(self, e2):
        """Cannot place a large disk on top of a small one on the shared pole."""
        s = e2.initial_state()
        s = e2.apply(s, Player.A, Move.lift("1a"))   # lifts disk 1
        s = e2.apply(s, Player.A, Move.place("2"))   # puts disk 1 on pole 2
        s = e2.apply(s, Player.B, Move.lift("1b"))   # B lifts disk 2
        # Disk 2 > disk 1, so placing 2 on top of 1 is illegal
        assert Move.place("2") not in e2.legal_moves(s, Player.B)
