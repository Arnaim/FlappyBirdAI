import random

# Difficulty settings per level:
# (speed, gap, spawn_interval_frames, max_gap_movement, spawn_jitter)
DIFFICULTY_LEVELS = [
    (5.0, 165, 52, 70, 0.20),    # Level 1: closer spacing, moderate vertical shifts
    (5.5, 158, 48, 95, 0.22),    # Level 2: tighter rhythm & wider height swings
    (6.0, 150, 45, 120, 0.25),   # Level 3: two pipes often visible, varied heights
    (6.5, 144, 42, 145, 0.25),   # Level 4: quick adjustments between pipes
    (7.2, 138, 40, 170, 0.28),   # Level 5: fast & packed
    (7.8, 132, 38, 195, 0.28),   # Level 6: high pressure, big vertical transitions
    (8.5, 126, 36, 220, 0.30),   # Level 7: very tight & demanding
]


def get_difficulty(pipe_count):
    """Return (speed, gap, spawn_interval, max_gap_movement, spawn_jitter) by pipe count."""
    if pipe_count < 3:
        idx = 0
    elif pipe_count < 6:
        idx = 1
    elif pipe_count < 10:
        idx = 2
    elif pipe_count < 14:
        idx = 3
    elif pipe_count < 20:
        idx = 4
    elif pipe_count < 26:
        idx = 5
    else:
        idx = 6

    return DIFFICULTY_LEVELS[idx]


def get_spawn_interval(base_interval, jitter=0.25):
    """Return a randomized frame interval between consecutive pipes."""
    low = max(22, int(base_interval * (1.0 - jitter)))
    high = max(low + 1, int(base_interval * (1.0 + jitter)))
    return random.randint(low, high)


def get_next_gap_y(previous_gap_y, max_gap_movement, min_y=110, max_y=560):
    """Choose the next pipe gap center across a wider vertical range."""
    step = 10

    if previous_gap_y is None:
        return random.randrange(min_y, max_y + 1, step)

    min_step = min(20, max_gap_movement)
    candidates = [
        y
        for y in range(min_y, max_y + 1, step)
        if min_step <= abs(y - previous_gap_y) <= max_gap_movement
    ]

    if not candidates:
        candidates = [
            y
            for y in range(min_y, max_y + 1, step)
            if y != previous_gap_y
        ]

    return random.choice(candidates)