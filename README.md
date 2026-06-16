# Hanoi Crossing

A two-player take on the Tower of Hanoi. Each player has their own Hanoi to solve, but they share the middle pole, so it's not just a race. You can park a disk on the shared pole, block the other player, or straight up steal a disk they left sitting there.

I started messing with this after reading some combinatorial game theory papers on competitive Hanoi (Chappelon and Matsuura have a nice one on a two-player version with a "no re-move" rule, and there are the usual CGT surveys on partizan games). None of them set it up quite like this, with a shared pole and each player only seeing half the board, so I figured I'd build an engine and see how it actually plays.

It's a small thing. Python, a game engine, two CLIs to drive it.

## The rules

Two players, A and B. Each has their own pole 1 and pole 3, and they share pole 2 in the middle:

```
        1a
        |
 1b -- [2] -- 3b
        |
        3a
```

A sees `1a`, `2`, `3a` and starts with the odd disks (1, 3, 5, …) on `1a`. B sees `1b`, `2`, `3b` and starts with the even disks (2, 4, 6, …) on `1b`. Normal Hanoi placement applies: a disk only goes on an empty pole or on a bigger one.

On your turn you do exactly one thing:

- `lift <pole>` — take the top disk off a pole you can see (only if your hand's empty)
- `place <pole>` — drop the disk you're holding onto a pole (has to fit)
- `skip` — pass

If you try something illegal, nothing happens and you've wasted your turn.

You win when your hand is empty and, of the poles you can see, only your goal pole (`3a` for A, `3b` for B) has disks on it. The catch is that pole 2 has to be clear too, and since both players want to use it, that's where most of the tension comes from.

## Decisions I had to make

The rules leave a few things open, so here's what I went with and why:

- **States are immutable.** `apply()` hands back a new `GameState` instead of mutating in place. Made replay and the BFS solver way easier to reason about, and tests don't step on each other.
- **Illegal moves just burn the turn**, they don't raise. That sounds lazy but it's deliberate, it means a replay file can include bad moves on purpose and still be a valid test fixture.
- **Win is checked for both players after every move**, not just whoever just went. B clearing pole 2 can hand A the win on B's turn, so you have to look at both.
- **If both somehow win at once, A wins.** It basically never happens, but I didn't want it undefined.
- **Turn order is passed in**, not assumed. The engine doesn't care if it's A, B, A, B or some lopsided sequence. You give it the order.
- **Disks are just ints, stacks are bottom-to-top lists.** So `stack[-1]` is the top and the whole Hanoi rule is `stack[-1] > disk`.
- **Replay files are YAML**, with rows like `[player, action, pole]`. Easy to read and diffs stay clean in git.

## Layout

```
src/hanoi_crossing/
├── models.py       # the data: Player, Action, Move, GameState
├── engine.py       # the engine: initial_state, legal_moves, apply, is_over
└── cli/
    ├── replay.py       # hanoi-replay: run a pre-recorded game
    └── random_play.py  # hanoi-random: both sides play random valid moves
tests/
└── test_engine.py
examples/
└── *.yaml          # a few sample games
```

`models.py` plus `engine.py` is about 150 lines. I wanted to keep the core tiny.

## Running it

```bash
uv sync
uv run pytest
```

### Replay a game

```bash
uv run hanoi-replay examples/n1_example.yaml
```

The file looks like this:

```yaml
n: 1
turns:
  - [A, lift, 1a]   # [player, action, pole]
  - [B, lift, 1b]
  - [A, place, 3a]
  - [A, skip]       # leave the pole off for skip
```

### Watch a random game

```bash
uv run hanoi-random             # n=2, up to 500 turns
uv run hanoi-random 3           # n=3
uv run hanoi-random 2 1000 42   # n=2, max 1000 turns, seed 42 so it's repeatable
```

## Stuff I read / left in

If you want to dig further: Chappelon and Matsuura's work on two-player Tower of Hanoi, and the CGT surveys on partizan games. There's also a `find_shortest_win()` in `engine.py` (BFS, uses `board_key()` as a transposition table) that I used to check shortest-win lengths against what the theory predicts. It's not wired into the CLIs, it was mostly for my own sanity checking.
