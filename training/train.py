
import copy
import json
import pickle
import random
from pathlib import Path

import neat
import pygame
from PIL import Image

from game.bird import Bird
from game.difficulty import (
    DIFFICULTY_LEVELS,
    get_next_gap_y,
    get_spawn_interval,
)
from game.pipe import Pipe


# Project paths
BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "assets").exists() and (BASE_DIR.parent / "assets").exists():
    BASE_DIR = BASE_DIR.parent

ASSETS_DIR = BASE_DIR / "assets"
CONFIG_PATH = BASE_DIR / "config" / "neat_config.txt"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"


# Game settings
WIDTH = 700
HEIGHT = 700
GROUND_HEIGHT = 50
GROUND_Y = HEIGHT - GROUND_HEIGHT

BIRD_X = 100
BIRD_Y = 300

MAX_GENERATION_FRAMES = 60 * 45
JUMP_THRESHOLD = 0.5

GENERATIONS = 50

# Fixed evaluation seeds: identical for every generation
EVALUATION_SEEDS = tuple(range(10000, 10010))

# Use the easiest existing difficulty for the baseline
TRAINING_DIFFICULTY = 0


# Best evaluated genome
BEST_GENOME = None
BEST_FITNESS = float("-inf")
BEST_EVALUATION_AVERAGE = -1
BEST_EVALUATION_SCORES = []
GENERATION_NUMBER = 0


def load_image(path):
    image = Image.open(path).convert("RGBA")
    return pygame.image.frombytes(
        image.tobytes(),
        image.size,
        "RGBA",
    )


def load_assets():
    pipe_image = load_image(ASSETS_DIR / "pipe.png")
    return pipe_image


def load_neat_config():
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(CONFIG_PATH),
    )


def create_pipe(pipe_image, gap_y, gap, speed):
    return Pipe(
        WIDTH,
        gap_y,
        pipe_image,
        GROUND_Y,
        gap=gap,
        speed=speed,
    )


def get_network_inputs(bird, pipes):
    next_pipe = next(
        (
            pipe for pipe in pipes
            if pipe.x + pipe.width >= bird.rect.centerx
        ),
        None,
    )

    if next_pipe is None:
        return (
            bird.rect.centery / HEIGHT,
            bird.velocity / 10.0,
            1.0,
            0.5,
        )

    distance = max(
        0.0,
        (next_pipe.x - bird.rect.centerx) / WIDTH,
    )

    gap_center = (
        next_pipe.top_rect.bottom + next_pipe.bottom_rect.top
    ) / 2

    return (
        bird.rect.centery / HEIGHT,
        bird.velocity / 10.0,
        distance,
        gap_center / HEIGHT,
    )


def evaluate_genome(genome, config, pipe_image, seeds):
    """Evaluate one genome against repeatable pipe sequences."""

    speed, gap, spawn_interval, max_gap_movement, spawn_jitter = (
        DIFFICULTY_LEVELS[TRAINING_DIFFICULTY]
    )

    results = []
    original_random_state = random.getstate()

    try:
        for seed in seeds:
            random.seed(seed)

            net = neat.nn.FeedForwardNetwork.create(genome, config)
            bird = Bird(BIRD_X, BIRD_Y)

            pipes = []
            passed_pipes = set()

            previous_gap_y = None
            spawn_counter = 0
            next_spawn_interval = get_spawn_interval(
                spawn_interval,
                spawn_jitter,
            )

            gap_y = get_next_gap_y(
                previous_gap_y,
                max_gap_movement,
            )
            previous_gap_y = gap_y

            pipes.append(
                create_pipe(pipe_image, gap_y, gap, speed)
            )

            score = 0
            frames = 0

            while frames < MAX_GENERATION_FRAMES:
                frames += 1
                spawn_counter += 1

                if spawn_counter >= next_spawn_interval:
                    spawn_counter = 0
                    next_spawn_interval = get_spawn_interval(
                        spawn_interval,
                        spawn_jitter,
                    )

                    gap_y = get_next_gap_y(
                        previous_gap_y,
                        max_gap_movement,
                    )
                    previous_gap_y = gap_y

                    pipes.append(
                        create_pipe(pipe_image, gap_y, gap, speed)
                    )

                for pipe in pipes:
                    pipe.move()

                pipes = [
                    pipe for pipe in pipes
                    if pipe.x + pipe.width > 0
                ]

                bird.move()

                inputs = get_network_inputs(bird, pipes)

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
                        and pipe not in passed_pipes
                    ):
                        score += 1
                        passed_pipes.add(pipe)

            results.append(score)

    finally:
        random.setstate(original_random_state)

    return results


def save_best_model():
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = SAVED_MODELS_DIR / "best_genome.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(BEST_GENOME, file)

    speed, gap, spawn_interval, max_gap_movement, spawn_jitter = (
        DIFFICULTY_LEVELS[TRAINING_DIFFICULTY]
    )

    metadata = {
        "best_fitness": BEST_FITNESS,
        "evaluation_average": BEST_EVALUATION_AVERAGE,
        "evaluation_scores": BEST_EVALUATION_SCORES,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "difficulty": {
            "speed": speed,
            "gap": gap,
            "spawn_interval": spawn_interval,
            "max_gap_movement": max_gap_movement,
            "spawn_jitter": spawn_jitter,
        },
    }

    metadata_path = SAVED_MODELS_DIR / "model_metadata.json"

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    print("Best model updated:", model_path)


def eval_genomes(genomes, config):
    global BEST_GENOME
    global BEST_FITNESS
    global BEST_EVALUATION_AVERAGE
    global BEST_EVALUATION_SCORES
    global GENERATION_NUMBER

    GENERATION_NUMBER += 1

    speed, gap, spawn_interval, max_gap_movement, spawn_jitter = (
        DIFFICULTY_LEVELS[TRAINING_DIFFICULTY]
    )

    print(f"\nGeneration {GENERATION_NUMBER}/{GENERATIONS}")
    print(
        f"Speed: {speed} | Gap: {gap} | "
        f"Spawn interval: {spawn_interval}"
    )

    nets = []
    birds = []
    genome_list = []
    passed_pipes = []

    for _, genome in genomes:
        genome.fitness = 0.0

        nets.append(
            neat.nn.FeedForwardNetwork.create(genome, config)
        )
        birds.append(Bird(BIRD_X, BIRD_Y))
        genome_list.append(genome)
        passed_pipes.append(set())

    pipe_image = load_assets()

    pipes = []
    spawn_counter = 0
    next_spawn_interval = get_spawn_interval(
        spawn_interval,
        spawn_jitter,
    )

    previous_gap_y = None
    gap_y = get_next_gap_y(
        previous_gap_y,
        max_gap_movement,
    )
    previous_gap_y = gap_y

    pipes.append(
        create_pipe(pipe_image, gap_y, gap, speed)
    )

    frame_count = 0

    # All birds share the same pipe sequence this generation.
    while birds and frame_count < MAX_GENERATION_FRAMES:
        frame_count += 1
        spawn_counter += 1

        if spawn_counter >= next_spawn_interval:
            spawn_counter = 0
            next_spawn_interval = get_spawn_interval(
                spawn_interval,
                spawn_jitter,
            )

            gap_y = get_next_gap_y(
                previous_gap_y,
                max_gap_movement,
            )
            previous_gap_y = gap_y

            pipes.append(
                create_pipe(pipe_image, gap_y, gap, speed)
            )

        for pipe in pipes:
            pipe.move()

        pipes = [
            pipe for pipe in pipes
            if pipe.x + pipe.width > 0
        ]

        # Update every living bird
        for i in range(len(birds) - 1, -1, -1):
            bird = birds[i]
            genome = genome_list[i]
            net = nets[i]

            bird.move()
            genome.fitness += 0.1

            inputs = get_network_inputs(bird, pipes)

            if net.activate(inputs)[0] > JUMP_THRESHOLD:
                bird.jump()

            collision = (
                bird.rect.bottom >= GROUND_Y
                or any(pipe.collide(bird) for pipe in pipes)
            )

            if collision:
                genome.fitness -= 2.0

                birds.pop(i)
                nets.pop(i)
                genome_list.pop(i)
                passed_pipes.pop(i)
                continue

            for pipe in pipes:
                if (
                    pipe.x + pipe.width < bird.rect.left
                    and pipe not in passed_pipes[i]
                ):
                    genome.fitness += 5.0
                    passed_pipes[i].add(pipe)

    # Find the strongest genome from this generation
    best_genome = max(
        (genome for _, genome in genomes),
        key=lambda genome: genome.fitness,
    )

    generation_best_fitness = best_genome.fitness

    print(f"Generation best fitness: {generation_best_fitness:.2f}")

    # Evaluate on the same 10 seeds every generation
    evaluation_scores = evaluate_genome(
        best_genome,
        config,
        pipe_image,
        EVALUATION_SEEDS,
    )

    average_score = sum(evaluation_scores) / len(evaluation_scores)

    print("Evaluation scores:", evaluation_scores)
    print(f"Evaluation average: {average_score:.2f} pipes")

    # Keep whichever genome performs best on the fixed benchmark
    if average_score > BEST_EVALUATION_AVERAGE:
        BEST_GENOME = copy.deepcopy(best_genome)
        BEST_FITNESS = generation_best_fitness
        BEST_EVALUATION_AVERAGE = average_score
        BEST_EVALUATION_SCORES = evaluation_scores.copy()

        print("New best evaluated genome!")
        save_best_model()


def main():
    print("Loading NEAT configuration...")

    pygame.init()

    try:
        config = load_neat_config()
        population = neat.Population(config)

        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())

        print("\nStarting fixed-difficulty NEAT training...")
        print("Population:", config.pop_size)
        print("Generations:", GENERATIONS)
        print("Difficulty level:", TRAINING_DIFFICULTY + 1)
        print("Evaluation seeds:", list(EVALUATION_SEEDS))
        print("Rendering: disabled")
        print("Press Ctrl+C in the terminal to stop.\n")

        population.run(eval_genomes, GENERATIONS)

        print("\nTraining completed!")
        print("Best fitness:", BEST_FITNESS)
        print("Best evaluation average:", BEST_EVALUATION_AVERAGE)
        print("Evaluation scores:", BEST_EVALUATION_SCORES)

        if BEST_GENOME is None:
            print("No evaluated genome was saved.")
        else:
            print("Best model is saved in saved_models/best_genome.pkl")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
