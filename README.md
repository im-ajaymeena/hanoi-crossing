# Hanoi Crossing

A two-player competitive Tower of Hanoi variant, implemented as a Python game engine with two CLI frontends.

## Game Rules

Two players (A and B) each have a private set of poles and share a middle pole (pole 2):

```
        1a
        |
 1b -- [2] -- 3b
        |
        3a
```

- **Player A** sees poles `1a`, `2`, `3a` and starts with odd disks (1, 3, 5, …) on `1a`.
- **Player B** sees poles `1b`, `2`, `3b` and starts with even disks (2, 4, 6, …) on `1b`.
- Standard Hanoi rule: a disk may only rest on an empty pole or a strictly larger disk.

On each turn a player does exactly **one** action:

| Action | Description |
|--------|-------------|
| `lift <pole>` | Pick up the top disk from a visible pole (hand must be empty) |
| `place <pole>` | Put the held disk onto a visible pole (must fit) |
| `skip` | Do nothing |

An **illegal** action wastes the turn — the board is unchanged.

**Win condition:** hand is empty AND among a player's visible poles, only their goal pole (`3a` for A, `3b` for B) has disks.

> The shared pole 2 must be clear for either player to win — this is the core strategic tension.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **State immutability** | `apply()` always returns a new `GameState`, never mutates the original | Makes replay, testing, and future AI search trivial |
| **Illegal move handling** | State unchanged, turn counter advances | Faithful to spec; replay files can include intentional illegal moves as test cases |
| **Win check timing** | After every `apply()`, both players are checked | A move by B could clear pole 2 and simultaneously trigger A's win condition |
| **Simultaneous wins** | First player found wins (A before B) | Rare edge case; documented as a tie-breaking convention |
| **Turn order** | Fully external — provided as a list, not assumed alternating | Matches the spec: "The engine must not assume any particular turn-order pattern" |
| **Disk representation** | Integer size; stacks stored bottom-to-top | Natural fit for the Hanoi rule (`stack[-1]` is always the top) |
| **Input format** | YAML with compact `[player, action, pole]` rows | Human-readable and diff-friendly for git history |

## Project Layout

```
src/hanoi_crossing/
├── models.py       # Pure data: Player, Action, Move, GameState
├── engine.py       # HanoiCrossing engine — initial_state, legal_moves, apply, is_over
└── cli/
    ├── replay.py       # hanoi-replay: replay pre-recorded moves
    └── random_play.py  # hanoi-random: both players make random valid moves
tests/
└── test_engine.py  # Engine unit tests (no CLI involved)
examples/
└── n1_example.yaml # N=1 walkthrough 
```

Core engine (`models.py` + `engine.py`): ~150 lines — well under the 500-line limit.

## Setup

```bash
uv sync
uv run pytest          # run tests
```

## Usage

### Replay mode

```bash
uv run hanoi-replay examples/n1_example.yaml
```

Replay file format:
```yaml
n: 1
turns:
  - [A, lift, 1a]   # [player, action, pole]
  - [B, lift, 1b]
  - [A, place, 3a]
  - [A, skip]       # pole omitted for skip
```

### Random-play mode

```bash
uv run hanoi-random             # n=2, up to 500 turns
uv run hanoi-random 3           # n=3
uv run hanoi-random 2 1000 42   # n=2, max 1000 turns, seed=42
```
