
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
FPS = 60
MAX_GENERATION_FRAMES = 60 * 45

GROUND_HEIGHT = 50
GROUND_Y = HEIGHT - GROUND_HEIGHT

BIRD_X = 100
BIRD_Y = 300

JUMP_THRESHOLD = 0.65


# Average pipes required before advancing difficulty
ADVANCE_THRESHOLDS = [6, 7, 8, 9, 10, 11]
ADVANCE_STREAK_REQUIRED = 2


# Fixed evaluation seeds.
# Every generation and every champion comparison uses these same sequences.
# Keep these separate from evaluate.py's unseen test seeds.
EVALUATION_SEEDS = tuple(range(10000, 10010))


# Training state
CURRICULUM_LEVEL = 0
ADVANCE_STREAK = 0

BEST_GENOME = None
BEST_FITNESS = float("-inf")
BEST_LEVEL = 0
BEST_EVALUATION_AVERAGE = -1


# Initialize Pygame
pygame.init()
pygame.font.init()

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird AI Training")

CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial", 24, bold=True)


def load_image(path):
    image = Image.open(path).convert("RGBA")
    return pygame.image.frombytes(
        image.tobytes(), image.size, "RGBA"
    )


def load_assets():
    background = load_image(ASSETS_DIR / "background.png")
    background = pygame.transform.smoothscale(
        background, (WIDTH, GROUND_Y)
    )

    ground = load_image(ASSETS_DIR / "ground.png")
    ground_width = max(
        1,
        int(
            ground.get_width()
            * GROUND_HEIGHT
            / ground.get_height()
        ),
    )
    ground = pygame.transform.smoothscale(
        ground, (ground_width, GROUND_HEIGHT)
    )

    pipe_image = load_image(ASSETS_DIR / "pipe.png")
    return background, ground, pipe_image


def load_neat_config():
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(CONFIG_PATH),
    )


def evaluate_genome(genome, config, difficulty_level, pipe_image, seeds):
    """Evaluate a genome on supplied repeatable pipe sequences."""
    speed, gap, spawn_interval, max_gap_movement, spawn_jitter = (
        DIFFICULTY_LEVELS[difficulty_level]
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
                spawn_interval, spawn_jitter
            )

            score = 0
            frames = 0

            gap_y = get_next_gap_y(
                previous_gap_y, max_gap_movement
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

            while frames < MAX_GENERATION_FRAMES:
                frames += 1
                spawn_counter += 1

                if spawn_counter >= next_spawn_interval:
                    spawn_counter = 0
                    next_spawn_interval = get_spawn_interval(
                        spawn_interval, spawn_jitter
                    )

                    gap_y = get_next_gap_y(
                        previous_gap_y, max_gap_movement
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
                        and pipe not in passed_pipes
                    ):
                        score += 1
                        passed_pipes.add(pipe)

            results.append(score)

    finally:
        random.setstate(original_random_state)

    return results


def eval_genomes(genomes, config):
    global CURRICULUM_LEVEL, ADVANCE_STREAK
    global BEST_GENOME, BEST_FITNESS, BEST_LEVEL
    global BEST_EVALUATION_AVERAGE

    speed, gap, spawn_interval, max_gap_movement, spawn_jitter = (
        DIFFICULTY_LEVELS[CURRICULUM_LEVEL]
    )

    print(
        f"\nTraining at difficulty level {CURRICULUM_LEVEL + 1}"
        f" | Speed: {speed}"
        f" | Gap: {gap}"
        f" | Interval: {spawn_interval} (±{int(spawn_jitter * 100)}%)"
    )

    nets = []
    birds = []
    genome_list = []
    passed_pipes = []

    for _, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(BIRD_X, BIRD_Y))
        genome_list.append(genome)
        passed_pipes.append(set())

    background, ground, pipe_image = load_assets()

    pipes = []
    ground_x = 0.0
    frame_count = 0
    spawn_counter = 0

    next_spawn_interval = get_spawn_interval(
        spawn_interval, spawn_jitter
    )

    previous_gap_y = None
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

    running = True

    while running and birds:
        CLOCK.tick(FPS)
        frame_count += 1
        spawn_counter += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                raise SystemExit

        if spawn_counter >= next_spawn_interval:
            spawn_counter = 0
            next_spawn_interval = get_spawn_interval(
                spawn_interval, spawn_jitter
            )

            gap_y = get_next_gap_y(
                previous_gap_y, max_gap_movement
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

        ground_x -= 4.0
        if ground_x <= -ground.get_width():
            ground_x += ground.get_width()

        for i in range(len(birds) - 1, -1, -1):
            bird = birds[i]
            genome = genome_list[i]
            net = nets[i]

            bird.move()
            genome.fitness += 0.1

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

            output = net.activate(inputs)[0]
            if output > JUMP_THRESHOLD:
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

        WIN.blit(background, (0, 0))

        for pipe in pipes:
            pipe.draw(WIN)

        for bird in birds:
            bird.draw(WIN)

        gx = int(ground_x)
        while gx < WIDTH:
            WIN.blit(ground, (gx, GROUND_Y))
            gx += ground.get_width()

        status = FONT.render(
            f"Alive: {len(birds)} / {len(genomes)}"
            f"   Level: {CURRICULUM_LEVEL + 1}"
            f"   Speed: {speed}",
            True,
            (255, 255, 255),
        )

        WIN.blit(status, (20, 20))
        pygame.display.update()

        if frame_count >= MAX_GENERATION_FRAMES:
            print("Generation time limit reached.")
            break

    best_genome = max(
        (genome for _, genome in genomes),
        key=lambda genome: genome.fitness,
    )
    best_fitness = best_genome.fitness

    print(f"Generation best fitness: {best_fitness:.2f}")

    # Evaluate every generation on the same fixed benchmark.
    evaluation_scores = evaluate_genome(
        best_genome,
        config,
        CURRICULUM_LEVEL,
        pipe_image,
        seeds=EVALUATION_SEEDS,
    )

    average_score = sum(evaluation_scores) / len(evaluation_scores)

    print(
        f"Evaluation seeds: {list(EVALUATION_SEEDS)}"
        f"\nEvaluation scores: {evaluation_scores}"
        f" | Average: {average_score:.1f} pipes"
    )

    if CURRICULUM_LEVEL < len(ADVANCE_THRESHOLDS):
        threshold = ADVANCE_THRESHOLDS[CURRICULUM_LEVEL]
        candidate_qualifies = average_score >= threshold
    else:
        candidate_qualifies = False

    if candidate_qualifies:
        replace_champion = False

        if BEST_GENOME is None or CURRICULUM_LEVEL > BEST_LEVEL:
            replace_champion = True

        elif CURRICULUM_LEVEL == BEST_LEVEL:
            champion_scores = evaluate_genome(
                BEST_GENOME,
                config,
                CURRICULUM_LEVEL,
                pipe_image,
                seeds=EVALUATION_SEEDS,
            )

            champion_average = (
                sum(champion_scores) / len(champion_scores)
            )

            print(
                f"Existing champion average: {champion_average:.1f}"
                f" | Candidate average: {average_score:.1f}"
            )

            if average_score > champion_average:
                replace_champion = True

        if replace_champion:
            BEST_GENOME = copy.deepcopy(best_genome)
            BEST_FITNESS = best_fitness
            BEST_LEVEL = CURRICULUM_LEVEL
            BEST_EVALUATION_AVERAGE = average_score

            print(
                "New proven champion!"
                f" Level: {BEST_LEVEL + 1}"
                f" | Average pipes: {BEST_EVALUATION_AVERAGE:.1f}"
            )

    if CURRICULUM_LEVEL < len(DIFFICULTY_LEVELS) - 1:
        if candidate_qualifies:
            ADVANCE_STREAK += 1

            print(
                f"Advance progress: "
                f"{ADVANCE_STREAK}/{ADVANCE_STREAK_REQUIRED}"
            )

            if ADVANCE_STREAK >= ADVANCE_STREAK_REQUIRED:
                CURRICULUM_LEVEL += 1
                ADVANCE_STREAK = 0

                print(
                    f"Difficulty increased to level "
                    f"{CURRICULUM_LEVEL + 1}!"
                )
        else:
            ADVANCE_STREAK = 0
            print("Difficulty unchanged. More practice needed.")


def main():
    print("Loading NEAT configuration...")
    config = load_neat_config()

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    print("\nStarting curriculum-based training...")
    print("Population:", config.pop_size)
    print("Generations: 50")
    print("Evaluation runs per generation champion: 10")
    print("Evaluation seeds:", list(EVALUATION_SEEDS))
    print("Evaluation seeds: fixed across generations")
    print("Final unseen testing: use evaluate.py")
    print("Press ESC or close the window to stop training.\n")

    winner = population.run(eval_genomes, 50)

    if BEST_GENOME is None:
        print(
            "\nNo genome met the evaluation threshold."
            "\nSaving the final NEAT winner as a fallback."
        )
        genome_to_save = winner
        level_to_save = 0
        fitness_to_save = winner.fitness
        average_to_save = 0
    else:
        genome_to_save = BEST_GENOME
        level_to_save = BEST_LEVEL
        fitness_to_save = BEST_FITNESS
        average_to_save = BEST_EVALUATION_AVERAGE

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SAVED_MODELS_DIR / "best_genome.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(genome_to_save, file)

    level_cfg = DIFFICULTY_LEVELS[level_to_save]

    metadata = {
        "best_fitness": fitness_to_save,
        "evaluation_average": average_to_save,
        "difficulty_level": level_to_save,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "difficulty": {
            "speed": level_cfg[0],
            "gap": level_cfg[1],
            "spawn_interval": level_cfg[2],
            "max_gap_movement": level_cfg[3],
            "spawn_jitter": level_cfg[4],
        },
    }

    metadata_path = SAVED_MODELS_DIR / "model_metadata.json"

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nTraining completed!")
    print("Best fitness:", fitness_to_save)
    print("Average evaluation score:", average_to_save)
    print("Winning difficulty level:", level_to_save + 1)
    print("Model saved to:", model_path)
    print("Metadata saved to:", metadata_path)

    pygame.quit()


if __name__ == "__main__":
    main()
