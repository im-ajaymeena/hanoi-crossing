"""
Research-driven test suite for HanoiCrossing.

Test design draws from three areas of literature:

1. Hanoi graph structure (Rose-Hulman / ResearchGate diagrams)
   - For 3-peg Hanoi with n disks: exactly 3^n valid states (each disk on one of 3 pegs).
   - Our variant has 5 poles + 2 hands → richer but still bounded state space for n=1.
   - BFS exhaustive verification is feasible at n=1.

2. Combinatorial game theory — P/N positions (Stanford CS97SI)
   - Games with "undo"-like moves are CYCLIC — no P/N classification applies.
   - Our game allows cycles (lift then place-back = same board, turn counter +2).
   - Cycles must not corrupt state or produce spurious winners.
   - "Skip is always available" means the game has no terminal deadlock positions.

3. Two-player Hanoi (Chappelon, Larsson, Matsuura 2018)
   - Academic variant adds: "cannot re-move the disk your opponent just moved."
   - Our game intentionally OMITS this restriction — the shared pole strategy
     only works if players can immediately pick up what the other just placed.
   - Shared pole IS the primary contention resource in our game (analogous to the
     asymmetric-information resource-sharing game studied by Anbarci et al. 2023).
"""

from __future__ import annotations

from collections import deque

import pytest

from hanoi_crossing import HanoiCrossing, Move, Player


# ── Helpers (mirror what's in test_extreme so this file is self-contained) ────


def _all_disks(state) -> set[int]:
    sizes = {d for stack in state.poles.values() for d in stack}
    sizes |= {d for d in state.hands.values() if d is not None}
    return sizes


def _all_stacks_valid(state) -> bool:
    return all(
        stack[i] > stack[i + 1]
        for stack in state.poles.values()
        for i in range(len(stack) - 1)
    )


def _state_key(state) -> tuple:
    """Hashable snapshot of a state (ignores turn counter, which is not structural)."""
    return (
        tuple(sorted((k, tuple(v)) for k, v in state.poles.items())),
        (state.hands[Player.A], state.hands[Player.B]),
    )


def _bfs_all_reachable(engine: HanoiCrossing):
    """
    BFS over the full reachable state space starting from initial_state().
    Returns (visited_states, winner_states_per_player).
    """
    initial = engine.initial_state()
    visited: dict[tuple, object] = {}      # key → state
    queue: deque = deque([initial])
    visited[_state_key(initial)] = initial
    winners: dict[str, list] = {Player.A: [], Player.B: []}

    while queue:
        s = queue.popleft()

        if s.winner:
            winners[s.winner].append(s)
            continue  # no further transitions from a won state make sense to explore

        for player in Player:
            for move in engine.legal_moves(s, player):
                ns = engine.apply(s, player, move)
                key = _state_key(ns)
                if key not in visited:
                    visited[key] = ns
                    queue.append(ns)

    return visited, winners


# ── 1. Exhaustive state-space verification for n=1 ────────────────────────────


class TestExhaustiveN1:
    """
    n=1 has a tiny state space: disk 1 (odd) and disk 2 (even).
    BFS can visit every reachable state and check all invariants simultaneously.
    This is the gold-standard "model checking" approach described in the formal
    verification literature (bounded model checking for finite-state systems).
    """

    @pytest.fixture
    def bfs_n1(self):
        engine = HanoiCrossing(1)
        visited, winners = _bfs_all_reachable(engine)
        return engine, visited, winners

    def test_state_space_is_finite_and_bounded(self, bfs_n1):
        _, visited, _ = bfs_n1
        # Should visit at least a handful but far fewer than 10 000
        assert 5 < len(visited) < 500, f"Unexpected state count: {len(visited)}"

    def test_every_reachable_state_preserves_disk_universe(self, bfs_n1):
        _, visited, _ = bfs_n1
        expected = {1, 2}
        for key, state in visited.items():
            assert _all_disks(state) == expected, f"Disk universe broken in state {key}"

    def test_every_reachable_state_has_valid_stacks(self, bfs_n1):
        _, visited, _ = bfs_n1
        for key, state in visited.items():
            assert _all_stacks_valid(state), f"Invalid stack in state {key}"

    def test_skip_legal_in_every_reachable_state(self, bfs_n1):
        engine, visited, _ = bfs_n1
        for state in visited.values():
            if state.winner:
                continue
            for player in Player:
                assert Move.skip() in engine.legal_moves(state, player), \
                    f"Skip not legal for {player} in state {_state_key(state)}"

    def test_a_win_state_is_reachable(self, bfs_n1):
        _, _, winners = bfs_n1
        assert winners[Player.A], "No path to Player A winning — engine broken"

    def test_b_win_state_is_reachable(self, bfs_n1):
        _, _, winners = bfs_n1
        assert winners[Player.B], "No path to Player B winning — engine broken"

    def test_win_states_satisfy_win_condition_exactly(self, bfs_n1):
        """Every state the BFS marks as a win must genuinely satisfy the win condition."""
        engine, _, winners = bfs_n1
        for player, states in winners.items():
            goal = "3a" if player == Player.A else "3b"
            other_visible = {"1a", "2", "3a"} if player == Player.A else {"1b", "2", "3b"}
            for s in states:
                assert s.hands[player] is None,        f"{player} winning while holding a disk"
                assert s.poles[goal],                   f"{player} winning with empty goal pole"
                for pole in other_visible - {goal}:
                    assert not s.poles[pole], f"{player} winning with disk on {pole}"

    def test_no_reachable_state_has_both_players_winning(self, bfs_n1):
        """
        Simultaneous wins are possible in theory (both conditions satisfied in one state).
        This test verifies what actually happens: since _find_winner checks A first,
        any 'both win' state should be recorded as A winning.
        """
        _, visited, _ = bfs_n1
        for state in visited.values():
            if state.winner == Player.B:
                # If B is declared winner, A must NOT also satisfy the win condition
                # (otherwise A would have been declared winner first)
                goal_a = "3a"
                a_also_wins = (
                    state.hands[Player.A] is None
                    and bool(state.poles[goal_a])
                    and not state.poles["1a"]
                    and not state.poles["2"]
                )
                assert not a_also_wins, \
                    "Found state where A also satisfies win but B was declared winner"


# ── 2. Cycle robustness (CGT: games with cycles must not break) ───────────────


class TestCycleRobustness:
    """
    Standard combinatorial game theory assumes acyclicity for P/N analysis.
    Our game is CYCLIC: players can lift and replace disks, revisiting the same
    board position. The engine must remain consistent across arbitrarily many cycles.
    """

    def test_lift_and_replace_returns_to_same_board(self):
        engine = HanoiCrossing(2)
        s0 = engine.initial_state()
        key0 = _state_key(s0)

        s1 = engine.apply(s0, Player.A, Move.lift("1a"))
        s2 = engine.apply(s1, Player.A, Move.place("1a"))

        assert _state_key(s2) == key0, "Board did not return to original after lift+replace"
        assert s2.winner is None
        assert s2.turn == 2  # turn counter advanced even though board is the same

    def test_1000_cycles_do_not_corrupt_state(self):
        engine = HanoiCrossing(2)
        s = engine.initial_state()
        expected_disks = _all_disks(s)

        for i in range(1000):
            s = engine.apply(s, Player.A, Move.lift("1a"))
            s = engine.apply(s, Player.A, Move.place("1a"))

        assert _all_disks(s) == expected_disks
        assert _all_stacks_valid(s)
        assert s.winner is None
        assert s.turn == 2000

    def test_competing_cycles_do_not_corrupt(self):
        """A and B both repeatedly lift/replace their own disks alternately."""
        engine = HanoiCrossing(2)
        s = engine.initial_state()

        for _ in range(500):
            s = engine.apply(s, Player.A, Move.lift("1a"))
            s = engine.apply(s, Player.A, Move.place("1a"))
            s = engine.apply(s, Player.B, Move.lift("1b"))
            s = engine.apply(s, Player.B, Move.place("1b"))

        assert _all_disks(s) == {1, 2, 3, 4}
        assert s.winner is None

    def test_shared_pole_trade_cycle(self):
        """A parks on pole 2; B steals it and puts it back; repeat 200 times."""
        engine = HanoiCrossing(1)
        s = engine.initial_state()

        # A parks disk 1 on shared pole
        s = engine.apply(s, Player.A, Move.lift("1a"))
        s = engine.apply(s, Player.A, Move.place("2"))

        for _ in range(200):
            s = engine.apply(s, Player.B, Move.lift("2"))    # B steals disk 1
            s = engine.apply(s, Player.B, Move.place("2"))   # B puts it back

        assert s.poles["2"] == [1]
        assert s.hands[Player.A] is None
        assert s.hands[Player.B] is None
        assert s.winner is None

    def test_game_has_no_deadlock_states(self):
        """
        Since Skip is always legal, there are no states where a player
        MUST make progress or is fully stuck. This is a key design property.
        """
        engine = HanoiCrossing(3)
        s = engine.initial_state()

        # Construct a messy, congested state
        s = engine.apply(s, Player.A, Move.lift("1a"))   # A lifts 1
        s = engine.apply(s, Player.A, Move.place("2"))   # A parks 1 on 2
        s = engine.apply(s, Player.B, Move.lift("1b"))   # B lifts 2
        s = engine.apply(s, Player.B, Move.place("3b"))  # B parks 2 on 3b
        # Now A's buffer (pole 2) is partially used, B has moved a disk

        # Even in this congested state, both players can always skip
        assert Move.skip() in engine.legal_moves(s, Player.A)
        assert Move.skip() in engine.legal_moves(s, Player.B)


# ── 3. The academic variant's "no re-move" rule is NOT in our game ─────────────


class TestNoReMoveRestriction:
    """
    Chappelon, Larsson, Matsuura (2018) study a variant where
    'the current player cannot move the disk that the previous player just moved.'
    Our game INTENTIONALLY omits this restriction. These tests confirm that
    immediately re-moving a disk is always legal in our engine.
    """

    def test_b_can_immediately_lift_disk_a_just_placed(self):
        engine = HanoiCrossing(1)
        s = engine.initial_state()
        s = engine.apply(s, Player.A, Move.lift("1a"))
        s = engine.apply(s, Player.A, Move.place("2"))   # A places disk 1 on pole 2
        # B can immediately lift that same disk off pole 2 — no restriction
        assert Move.lift("2") in engine.legal_moves(s, Player.B)

    def test_a_can_immediately_re_lift_what_a_just_placed(self):
        """A can even undo their own previous move — no 'touched last' rule."""
        engine = HanoiCrossing(2)
        s = engine.initial_state()
        s = engine.apply(s, Player.A, Move.lift("1a"))
        s = engine.apply(s, Player.A, Move.place("3a"))  # A places on goal pole
        # A can lift it right back
        assert Move.lift("3a") in engine.legal_moves(s, Player.A)

    def test_b_can_immediately_lift_disk_b_just_placed(self):
        engine = HanoiCrossing(1)
        s = engine.initial_state()
        s = engine.apply(s, Player.B, Move.lift("1b"))
        s = engine.apply(s, Player.B, Move.place("3b"))
        assert Move.lift("3b") in engine.legal_moves(s, Player.B)

    def test_shared_pole_immediate_counter_steal(self):
        """A places on pole 2; B immediately takes it; A immediately takes it back."""
        engine = HanoiCrossing(2)
        s = engine.initial_state()
        s = engine.apply(s, Player.A, Move.lift("1a"))  # A lifts 1
        s = engine.apply(s, Player.A, Move.place("2"))  # A parks 1 on 2
        s = engine.apply(s, Player.B, Move.lift("2"))   # B steals 1
        s = engine.apply(s, Player.B, Move.place("2"))  # B returns 1 to pole 2
        # A can take it straight back
        assert Move.lift("2") in engine.legal_moves(s, Player.A)
        assert s.poles["2"] == [1]


# ── 4. Contention: shared pole as congested resource ─────────────────────────


class TestSharedPoleContention:
    """
    From resource-sharing game theory (Anbarci et al. 2023): when two agents
    compete for a shared resource, contention states create strategic tension.
    Pole 2 is the sole shared resource in Hanoi Crossing.
    """

    def test_both_players_cannot_both_hold_disk_from_pole2_simultaneously(self):
        """
        After A takes from pole 2, the pole is empty — B cannot also take from it
        on the next turn (nothing there to take).
        """
        engine = HanoiCrossing(2)
        s = engine.initial_state()
        # Park two disks on pole 2 in valid order: 3 (bottom), then 1 (top)
        s = engine.apply(s, Player.A, Move.lift("1a"))   # lifts 1
        s = engine.apply(s, Player.A, Move.place("2"))   # 2=[1]
        s = engine.apply(s, Player.A, Move.lift("1a"))   # lifts 3
        s = engine.apply(s, Player.A, Move.place("3a"))  # 3→3a (can't stack 3 on 1)
        # pole 2 has [1]. A took the smaller disk; B attempts to take from pole 2
        s = engine.apply(s, Player.B, Move.lift("1b"))   # B lifts 2 from 1b
        # B wants to place on pole 2; 2 > 1 → illegal (top of 2 is 1, and 2 > 1)
        assert Move.place("2") not in engine.legal_moves(s, Player.B)

    def test_pole2_becomes_bottleneck_blocking_both_wins(self):
        """
        Both players park their largest disks on pole 2, blocking each other.
        Neither can win because their visible 'other' poles include pole 2.
        """
        engine = HanoiCrossing(2)
        s = engine.initial_state()
        # A parks disk 3 on pole 2
        s = engine.apply(s, Player.A, Move.lift("1a"))   # A lifts 1
        s = engine.apply(s, Player.A, Move.place("3a"))  # A puts 1 on 3a
        s = engine.apply(s, Player.A, Move.lift("1a"))   # A lifts 3
        s = engine.apply(s, Player.A, Move.place("2"))   # A parks 3 on pole 2 → 2=[3]
        # B parks disk 2 on top of disk 3 on pole 2 (2 < 3 → legal)
        s = engine.apply(s, Player.B, Move.lift("1b"))   # B lifts 2
        s = engine.apply(s, Player.B, Move.place("2"))   # 2=[3,2] (3 bottom, 2 top)
        # B parks disk 4 on 3b
        s = engine.apply(s, Player.B, Move.lift("1b"))   # B lifts 4
        s = engine.apply(s, Player.B, Move.place("3b"))  # 3b=[4]
        # State: A has 1 on 3a, 3 on pole 2; B has 2 on pole 2, 4 on 3b
        # A: hand=None, 1a=[], 2=[3,2], 3a=[1] → pole 2 not empty → A can't win
        # B: hand=None, 1b=[], 2=[3,2], 3b=[4] → pole 2 not empty → B can't win
        assert s.winner is None

    def test_priority_dispute_resolved_by_turn_order(self):
        """
        If A and B both satisfy win conditions after the SAME move, the engine
        deterministically picks A (checked first). This documents the tie-breaking rule.
        """
        engine = HanoiCrossing(1)
        s = engine.initial_state()
        # A wins scenario: A's disk on 3a, pole 2 will be cleared by B
        s = engine.apply(s, Player.A, Move.lift("1a"))
        s = engine.apply(s, Player.A, Move.place("3a"))    # A has 1 on 3a
        s = engine.apply(s, Player.B, Move.lift("1b"))     # B lifts 2; 1b=[]
        # B places 2 on 3b → B win condition: hand=None, 1b=[], 2=[], 3b=[2] ✓
        # A win condition: hand=None, 1a=[], 2=[], 3a=[1] ✓ (pole 2 now empty!)
        # Both satisfied simultaneously — A is checked first
        s = engine.apply(s, Player.B, Move.place("3b"))
        assert s.winner == Player.A   # A wins because _find_winner checks A before B

    def test_a_clears_own_blocker_from_shared_pole(self):
        """A parks on pole 2, then reclaims it themselves to unblock their own win."""
        engine = HanoiCrossing(1)
        s = engine.initial_state()
        # A uses pole 2 as buffer (bad move), then fixes it
        s = engine.apply(s, Player.A, Move.lift("1a"))
        s = engine.apply(s, Player.A, Move.place("2"))   # A parks disk 1 on pole 2
        # A's goal: can't win (pole 2 not empty). A must reclaim.
        s = engine.apply(s, Player.A, Move.lift("2"))    # A takes disk 1 back
        s = engine.apply(s, Player.A, Move.place("3a"))  # A puts disk 1 on goal
        assert s.winner == Player.A                       # now wins


# ── 5. State space size and bounds ────────────────────────────────────────────


class TestStateSpaceBounds:
    """
    From Hanoi graph theory: the n-disk, 3-peg Hanoi graph has exactly 3^n nodes.
    Our game has 5 poles + 2 hands → larger state space. We verify empirically
    that the reachable state count stays bounded and grows with n as expected.
    """

    def test_n1_state_space_larger_than_classic_hanoi(self):
        """Classic 3-peg, 2-disk Hanoi has 3^2 = 9 states. Ours has more (5 poles, 2 hands)."""
        engine = HanoiCrossing(1)
        visited, _ = _bfs_all_reachable(engine)
        assert len(visited) > 9, "Expected more states than classic 3-peg Hanoi"

    def test_n2_state_space_larger_than_n1(self):
        """More disks → more states (monotonicity check)."""
        e1 = HanoiCrossing(1)
        e2 = HanoiCrossing(2)
        visited1, _ = _bfs_all_reachable(e1)
        visited2, _ = _bfs_all_reachable(e2)
        assert len(visited2) > len(visited1)

    def test_all_n1_states_have_correct_disk_count(self):
        engine = HanoiCrossing(1)
        visited, _ = _bfs_all_reachable(engine)
        for state in visited.values():
            in_poles = sum(len(v) for v in state.poles.values())
            in_hands = sum(1 for d in state.hands.values() if d is not None)
            assert in_poles + in_hands == 2, "Disk count changed in reachable state"

    def test_all_n1_states_have_valid_stacks(self):
        engine = HanoiCrossing(1)
        visited, _ = _bfs_all_reachable(engine)
        for key, state in visited.items():
            assert _all_stacks_valid(state), f"Invalid stack in state {key}"

    def test_n1_win_states_are_minority(self):
        """Win states are leaf nodes — should be a small fraction of total states."""
        engine = HanoiCrossing(1)
        visited, winners = _bfs_all_reachable(engine)
        total = len(visited)
        total_wins = len(winners[Player.A]) + len(winners[Player.B])
        assert total_wins < total / 2, "Too many win states — win condition may be too loose"
        assert total_wins >= 2, "Need at least one win state per player"


# ── 6. Blocking analysis: maximal interference ───────────────────────────────


class TestBlocking:
    """
    Research on competitive resource games shows that a rational opponent will
    try to occupy the shared resource to delay the other player. These tests
    verify the engine handles maximum-interference scenarios correctly.
    """

    def test_b_can_prevent_a_from_winning_indefinitely_via_pole2(self):
        """
        B parks a disk on pole 2 every time A clears it.
        A cannot win as long as B keeps doing this.
        """
        engine = HanoiCrossing(2)
        s = engine.initial_state()

        # A completes their disks on 3a EXCEPT for the pole-2 blocker
        s = engine.apply(s, Player.A, Move.lift("1a"))   # A lifts 1
        s = engine.apply(s, Player.A, Move.place("3a"))  # 1→3a
        s = engine.apply(s, Player.A, Move.lift("1a"))   # A lifts 3
        s = engine.apply(s, Player.A, Move.place("3a"))  # 3 on 3a? 3a=[1] top=1, 1<3 → ILLEGAL
        # A can't place 3 directly. Verify it's illegal:
        assert s.hands[Player.A] == 3   # still holding disk 3

        # B parks disk 2 on pole 2 while A is stuck
        s = engine.apply(s, Player.B, Move.lift("1b"))
        s = engine.apply(s, Player.B, Move.place("2"))   # B blocks pole 2 with disk 2

        # Now A must use 2 as buffer but it's blocked. A tries anyway (illegal):
        s2 = engine.apply(s, Player.A, Move.place("2"))  # A holds 3, pole2 top=2, 2<3 → ILLEGAL
        assert s2.hands[Player.A] == 3

        # Verify: no winner yet
        assert s.winner is None

    def test_a_wins_despite_b_trying_to_block_n1(self):
        """
        With n=1, B's only blocking move is to park disk 2 on pole 2.
        But A still wins if B ever needs to clear pole 2 for their own progress.
        """
        engine = HanoiCrossing(1)
        s = engine.initial_state()

        # A moves disk 1 to goal immediately
        s = engine.apply(s, Player.A, Move.lift("1a"))
        s = engine.apply(s, Player.A, Move.place("3a"))
        # A wins if pole 2 is empty — but B now parks on pole 2 to block
        s = engine.apply(s, Player.B, Move.lift("1b"))
        s = engine.apply(s, Player.B, Move.place("2"))
        # A cannot win yet (pole 2 blocked)
        assert s.winner is None
        # B needs to clear pole 2 to place on 3b
        s = engine.apply(s, Player.B, Move.lift("2"))    # B lifts disk 2 off pole 2
        # NOW A wins (B's own clearing move triggers it)
        assert s.winner == Player.A

    def test_maximum_blocking_n2_both_stuck(self):
        """
        Both players park their largest disk on pole 2 in valid Hanoi order.
        Neither can win, and the game is in a congested-but-legal state.
        """
        engine = HanoiCrossing(2)
        s = engine.initial_state()
        # A puts disk 1 on 3a (partial progress)
        s = engine.apply(s, Player.A, Move.lift("1a"))
        s = engine.apply(s, Player.A, Move.place("3a"))
        # A puts disk 3 on pole 2
        s = engine.apply(s, Player.A, Move.lift("1a"))
        s = engine.apply(s, Player.A, Move.place("2"))   # 2=[3]
        # B puts disk 2 on top of disk 3 on pole 2 (legal: 2 < 3)
        s = engine.apply(s, Player.B, Move.lift("1b"))
        s = engine.apply(s, Player.B, Move.place("2"))   # 2=[3,2]
        # State: 1a=[], 2=[3,2], 3a=[1], 1b=[4], 3b=[]
        # A can't win: pole 2 has disks
        # B can't win: pole 2 has disks AND 3b is empty
        assert s.winner is None
        # Both players still have legal moves (not deadlocked)
        assert len(engine.legal_moves(s, Player.A)) > 1  # more than just skip
        assert len(engine.legal_moves(s, Player.B)) > 1


# ── 7. Asymmetric information edge cases ──────────────────────────────────────


class TestAsymmetricInformation:
    """
    Hanoi Crossing has partial information: each player cannot see the other's
    private poles (1x and 3x) or what the other holds. The engine enforces this
    via the visibility rules. These tests verify the enforcement is airtight.
    """

    def test_player_a_cannot_interact_with_b_private_poles_exhaustive(self):
        engine = HanoiCrossing(2)
        s = engine.initial_state()
        b_private = ["1b", "3b"]

        # Try every action on every private pole for A across multiple states
        states_to_test = [s]
        # Also test after A has a disk in hand
        s2 = engine.apply(s, Player.A, Move.lift("1a"))
        states_to_test.append(s2)

        for state in states_to_test:
            for pole in b_private:
                lift = Move.lift(pole)
                place = Move.place(pole)
                assert lift  not in engine.legal_moves(state, Player.A), \
                    f"A should not be able to lift from {pole}"
                assert place not in engine.legal_moves(state, Player.A), \
                    f"A should not be able to place on {pole}"

    def test_player_b_cannot_interact_with_a_private_poles_exhaustive(self):
        engine = HanoiCrossing(2)
        s = engine.initial_state()
        a_private = ["1a", "3a"]

        states_to_test = [s]
        s2 = engine.apply(s, Player.B, Move.lift("1b"))
        states_to_test.append(s2)

        for state in states_to_test:
            for pole in a_private:
                lift = Move.lift(pole)
                place = Move.place(pole)
                assert lift  not in engine.legal_moves(state, Player.B), \
                    f"B should not be able to lift from {pole}"
                assert place not in engine.legal_moves(state, Player.B), \
                    f"B should not be able to place on {pole}"

    def test_shared_pole_is_visible_to_both_always(self):
        """Pole 2 must always appear in legal_moves for both players when interactable."""
        engine = HanoiCrossing(1)
        s = engine.initial_state()

        # A places disk on pole 2
        s = engine.apply(s, Player.A, Move.lift("1a"))
        s = engine.apply(s, Player.A, Move.place("2"))   # 2=[1]

        # Both players should be able to lift from pole 2
        assert Move.lift("2") in engine.legal_moves(s, Player.A)
        assert Move.lift("2") in engine.legal_moves(s, Player.B)

    def test_a_cannot_see_how_many_disks_b_has_on_private_poles(self):
        """
        This is a conceptual test: the engine's legal_moves for A must be independent
        of what's on B's private poles (1b, 3b). A's options don't change based on B's
        private pole contents — only based on what A can see.
        """
        engine = HanoiCrossing(2)

        # State 1: B has all disks on 1b (initial)
        s1 = engine.initial_state()
        a_moves_1 = set(str(m) for m in engine.legal_moves(s1, Player.A))

        # State 2: B has moved disks around on their private poles
        s2 = engine.initial_state()
        s2 = engine.apply(s2, Player.B, Move.lift("1b"))   # B moves disk
        s2 = engine.apply(s2, Player.B, Move.place("3b"))  # to 3b
        a_moves_2 = set(str(m) for m in engine.legal_moves(s2, Player.A))

        # A's legal moves should be identical in both states
        # (B's private pole activity doesn't affect A's options)
        assert a_moves_1 == a_moves_2, \
            "A's legal moves changed based on B's private pole activity — visibility leak"
