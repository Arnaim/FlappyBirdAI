
import json
import pickle
import random
from pathlib import Path

import neat
import pygame

from game.bird import Bird
from game.difficulty import get_next_gap_y, get_spawn_interval
from game.pipe import Pipe


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_models" / "best_genome.pkl"
METADATA_PATH = BASE_DIR / "saved_models" / "model_metadata.json"
CONFIG_PATH = BASE_DIR / "config" / "neat_config.txt"
ASSETS_DIR = BASE_DIR / "assets"

# Game settings (must match training)
WIDTH = 700
HEIGHT = 700
FPS = 60
MAX_FRAMES = FPS * 45

GROUND_HEIGHT = 50
GROUND_Y = HEIGHT - GROUND_HEIGHT

BIRD_X = 100
BIRD_Y = 300

JUMP_THRESHOLD = 0.5
TEST_RUNS = 20
TEST_SEEDS = range(2000, 2020)


def load_image(path):
    from PIL import Image

    image = Image.open(path).convert("RGBA")
    return pygame.image.frombytes(
        image.tobytes(), image.size, "RGBA"
    )


def load_config():
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(CONFIG_PATH),
    )


def evaluate_one(net, pipe_image, difficulty, seed):
    speed, gap, spawn_interval, max_gap_movement, spawn_jitter = difficulty

    random.seed(seed)

    bird = Bird(BIRD_X, BIRD_Y)
    pipes = []

    previous_gap_y = None
    spawn_counter = 0

    next_spawn_interval = get_spawn_interval(
        spawn_interval, spawn_jitter
    )

    gap_y = get_next_gap_y(previous_gap_y, max_gap_movement)
    previous_gap_y = gap_y

    pipes.append(
        Pipe(
            WIDTH,
            gap_y,
            pipe_image,
            GROUND_Y,
            gap=gap,
            speed=speed,
        )
    )

    score = 0
    frames = 0

    while frames < MAX_FRAMES:
        frames += 1
        spawn_counter += 1

        if spawn_counter >= next_spawn_interval:
            spawn_counter = 0

            next_spawn_interval = get_spawn_interval(
                spawn_interval, spawn_jitter
            )

            gap_y = get_next_gap_y(
                previous_gap_y,
                max_gap_movement,
            )
            previous_gap_y = gap_y

            pipes.append(
                Pipe(
                    WIDTH,
                    gap_y,
                    pipe_image,
                    GROUND_Y,
                    gap=gap,
                    speed=speed,
                )
            )

        for pipe in pipes:
            pipe.move()

        pipes = [
            pipe for pipe in pipes
            if pipe.x + pipe.width > 0
        ]

        bird.move()

        next_pipe = next(
            (
                pipe for pipe in pipes
                if pipe.x + pipe.width >= bird.rect.centerx
            ),
            None,
        )

        if next_pipe is not None:
            distance = max(
                0.0,
                (next_pipe.x - bird.rect.centerx) / WIDTH,
            )

            gap_center = (
                next_pipe.top_rect.bottom
                + next_pipe.bottom_rect.top
            ) / 2

            inputs = (
                bird.rect.centery / HEIGHT,
                bird.velocity / 10.0,
                distance,
                gap_center / HEIGHT,
            )
        else:
            inputs = (
                bird.rect.centery / HEIGHT,
                bird.velocity / 10.0,
                1.0,
                0.5,
            )

        if net.activate(inputs)[0] > JUMP_THRESHOLD:
            bird.jump()

        collision = (
            bird.rect.bottom >= GROUND_Y
            or any(pipe.collide(bird) for pipe in pipes)
        )

        if collision:
            break

        for pipe in pipes:
            if (
                pipe.x + pipe.width < bird.rect.left
                and not pipe.passed
            ):
                pipe.passed = True
                score += 1

    return score, frames


def main():
    pygame.init()

    print("Loading champion model...")

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    difficulty_data = metadata["difficulty"]

    difficulty = (
        difficulty_data["speed"],
        difficulty_data["gap"],
        difficulty_data["spawn_interval"],
        difficulty_data["max_gap_movement"],
        difficulty_data["spawn_jitter"],
    )

    config = load_config()

    with open(MODEL_PATH, "rb") as file:
        genome = pickle.load(file)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    pipe_image = load_image(ASSETS_DIR / "pipe.png")

    print("Model loaded.")
    print("Difficulty level: 1 (Fixed baseline)")
    print("Testing on 20 unseen seeds...\n")

    results = []
    original_random_state = random.getstate()

    try:
        for index, seed in enumerate(TEST_SEEDS, start=1):
            score, frames = evaluate_one(
                net,
                pipe_image,
                difficulty,
                seed,
            )

            results.append(score)

            print(
                f"Test {index:02}/{TEST_RUNS}"
                f" | Seed: {seed}"
                f" | Pipes: {score}"
                f" | Survival: {frames / FPS:.1f}s"
            )

    finally:
        random.setstate(original_random_state)
        pygame.quit()

    average = sum(results) / len(results)

    print("\n========== FINAL RESULTS ==========")
    print("Individual scores:", results)
    print(f"Average pipes: {average:.2f}")
    print("Minimum pipes:", min(results))
    print("Maximum pipes:", max(results))
    print("Total tests:", len(results))


if __name__ == "__main__":
    main()
