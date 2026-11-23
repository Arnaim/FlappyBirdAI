
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
GENERATIONS_PER_LEVEL = 10

# Fixed evaluation seeds
EVALUATION_SEEDS = tuple(range(10000, 10010))

# Five progressive stages: Levels 1 to 5
TRAINING_LEVELS = [0, 1, 2, 3, 4]


# Training state
GENERATION_NUMBER = 0
CURRENT_DIFFICULTY = 0

# Track the best evaluation separately for each level
STAGE_BEST_AVERAGES = {
    level: -1 for level in TRAINING_LEVELS
}


def load_image(path):
    image = Image.open(path).convert("RGBA")
    return pygame.image.frombytes(
        image.tobytes(),
        image.size,
        "RGBA",
    )


def load_assets():
    return load_image(ASSETS_DIR / "pipe.png")


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


def get_current_difficulty():
    return DIFFICULTY_LEVELS[CURRENT_DIFFICULTY]


def evaluate_genome(genome, config, pipe_image, seeds):
    """Evaluate one genome on repeatable pipe sequences."""

    speed, gap, spawn_interval, max_gap_movement, spawn_jitter = (
        get_current_difficulty()
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


def save_stage_model(genome, fitness, average, scores):
    """Save the best model for the current difficulty stage."""

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    level = CURRENT_DIFFICULTY + 1

    model_path = SAVED_MODELS_DIR / (
        f"progressive_level_{level}.pkl"
    )

    metadata_path = SAVED_MODELS_DIR / (
        f"progressive_level_{level}_metadata.json"
    )

    with open(model_path, "wb") as file:
        pickle.dump(genome, file)

    speed, gap, spawn_interval, max_gap_movement, spawn_jitter = (
        get_current_difficulty()
    )

    metadata = {
        "training_type": "progressive",
        "level": level,
        "generation": GENERATION_NUMBER,
        "best_fitness": fitness,
        "evaluation_average": average,
        "evaluation_scores": scores,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "difficulty": {
            "speed": speed,
            "gap": gap,
            "spawn_interval": spawn_interval,
            "max_gap_movement": max_gap_movement,
            "spawn_jitter": spawn_jitter,
        },
    }

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    # The Level 5 champion is also the final progressive model.
    if level == 5:
        final_model_path = (
            SAVED_MODELS_DIR / "best_genome_progressive.pkl"
        )
        final_metadata_path = (
            SAVED_MODELS_DIR / "model_metadata_progressive.json"
        )

        with open(final_model_path, "wb") as file:
            pickle.dump(genome, file)

        with open(final_metadata_path, "w") as file:
            json.dump(metadata, file, indent=4)

    print("Saved stage model:", model_path)


def eval_genomes(genomes, config):
    global GENERATION_NUMBER
    global CURRENT_DIFFICULTY

    GENERATION_NUMBER += 1

    # Increase difficulty every 10 generations.
    stage_index = min(
        (GENERATION_NUMBER - 1) // GENERATIONS_PER_LEVEL,
        len(TRAINING_LEVELS) - 1,
    )

    CURRENT_DIFFICULTY = TRAINING_LEVELS[stage_index]

    speed, gap, spawn_interval, max_gap_movement, spawn_jitter = (
        get_current_difficulty()
    )

    level = CURRENT_DIFFICULTY + 1

    print(f"\nGeneration {GENERATION_NUMBER}/{GENERATIONS}")
    print(f"Training difficulty: Level {level}")
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

    # All birds share the same pipe sequence.
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

        # Update every living bird.
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

    # Find the strongest genome in this generation.
    best_genome = max(
        (genome for _, genome in genomes),
        key=lambda genome: genome.fitness,
    )

    generation_best_fitness = best_genome.fitness

    print(
        f"Generation best fitness: "
        f"{generation_best_fitness:.2f}"
    )

    # Evaluate the generation champion at the current level.
    evaluation_scores = evaluate_genome(
        best_genome,
        config,
        pipe_image,
        EVALUATION_SEEDS,
    )

    average_score = (
        sum(evaluation_scores) / len(evaluation_scores)
    )

    print("Evaluation scores:", evaluation_scores)
    print(f"Evaluation average: {average_score:.2f} pipes")

    # Compare only against previous generations at this same level.
    if average_score > STAGE_BEST_AVERAGES[CURRENT_DIFFICULTY]:
        STAGE_BEST_AVERAGES[CURRENT_DIFFICULTY] = average_score

        save_stage_model(
            copy.deepcopy(best_genome),
            generation_best_fitness,
            average_score,
            evaluation_scores.copy(),
        )

        print(f"New best model for Level {level}!")


def main():
    print("Loading NEAT configuration...")

    pygame.init()

    try:
        config = load_neat_config()
        population = neat.Population(config)

        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())

        print("\nStarting progressive NEAT training...")
        print("Population:", config.pop_size)
        print("Generations:", GENERATIONS)
        print("Generations per level:", GENERATIONS_PER_LEVEL)
        print("Training levels: 1 to 5")
        print("Evaluation seeds:", list(EVALUATION_SEEDS))
        print("Rendering: disabled")
        print("Press Ctrl+C in the terminal to stop.\n")

        population.run(eval_genomes, GENERATIONS)

        print("\nProgressive training completed!")

        print("\nBest evaluation averages:")
        for level in TRAINING_LEVELS:
            average = STAGE_BEST_AVERAGES[level]
            print(f"Level {level + 1}: {average:.2f} pipes")

        print(
            "\nLevel 5 model:",
            "saved_models/best_genome_progressive.pkl",
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
