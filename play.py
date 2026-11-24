
import json
import pickle
from pathlib import Path

import neat
import pygame
from PIL import Image

from game.bird import Bird
from game.pipe import Pipe
from game.difficulty import (
    get_next_gap_y,
    get_spawn_interval,
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
CONFIG_PATH = BASE_DIR / "config" / "neat_config.txt"

MODEL_PATH = BASE_DIR / "saved_models" / "best_genome_progressive.pkl"
METADATA_PATH = BASE_DIR / "saved_models" / "model_metadata_progressive.json"


# Game settings
WIDTH = 700
HEIGHT = 700
FPS = 60

GROUND_HEIGHT = 50
GROUND_Y = HEIGHT - GROUND_HEIGHT

BIRD_X = 100
BIRD_Y = 300

MAX_GENERATION_FRAMES = 60 * 45
JUMP_THRESHOLD = 0.5


pygame.init()
pygame.font.init()

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird AI - Progressive Level 5")

CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial", 24, bold=True)


def load_image(path):
    image = Image.open(path).convert("RGBA")

    return pygame.image.frombytes(
        image.tobytes(),
        image.size,
        "RGBA"
    )


def load_assets():
    background = load_image(ASSETS_DIR / "background.png")

    background = pygame.transform.smoothscale(
        background,
        (WIDTH, GROUND_Y)
    )

    ground = load_image(ASSETS_DIR / "ground.png")

    ground_width = max(
        1,
        int(
            ground.get_width()
            * GROUND_HEIGHT
            / ground.get_height()
        )
    )

    ground = pygame.transform.smoothscale(
        ground,
        (ground_width, GROUND_HEIGHT)
    )

    pipe_image = load_image(ASSETS_DIR / "pipe.png")

    return background, ground, pipe_image


def load_neat_config():
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(CONFIG_PATH)
    )


def main():
    print("Loading progressive AI model...")

    # Load model metadata.
    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    difficulty = metadata["difficulty"]

    test_level = 5

    speed = difficulty["speed"]
    gap = difficulty["gap"]
    spawn_interval = difficulty["spawn_interval"]
    max_gap_movement = difficulty["max_gap_movement"]
    spawn_jitter = difficulty["spawn_jitter"]

    # Load NEAT configuration and trained genome.
    config = load_neat_config()

    with open(MODEL_PATH, "rb") as file:
        genome = pickle.load(file)

    net = neat.nn.FeedForwardNetwork.create(
        genome,
        config
    )

    print("Model loaded.")
    print(f"Best fitness: {metadata['best_fitness']:.2f}")
    print(f"Testing difficulty level: {test_level}")
    print(f"Speed: {speed}")
    print(f"Gap: {gap}")
    print(f"Spawn interval: {spawn_interval}")
    print(f"Spawn jitter: {spawn_jitter}")
    print(f"Maximum gap movement: {max_gap_movement}")
    print("Press ESC to quit.\n")

    bird = Bird(BIRD_X, BIRD_Y)

    background, ground, pipe_image = load_assets()

    pipes = []

    previous_gap_y = None

    gap_y = get_next_gap_y(
        previous_gap_y,
        max_gap_movement
    )

    previous_gap_y = gap_y

    pipes.append(
        Pipe(
            WIDTH,
            gap_y,
            pipe_image,
            GROUND_Y,
            gap=gap,
            speed=speed
        )
    )

    pipe_count = 1
    spawn_counter = 0

    next_spawn_interval = get_spawn_interval(
        spawn_interval,
        spawn_jitter
    )

    ground_x = 0.0
    score = 0
    frame_count = 0

    running = True
    crashed = False

    while running and frame_count < MAX_GENERATION_FRAMES:
        CLOCK.tick(FPS)

        frame_count += 1
        spawn_counter += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                running = False

        if not running:
            break

        # Spawn pipes.
        if spawn_counter >= next_spawn_interval:
            spawn_counter = 0

            next_spawn_interval = get_spawn_interval(
                spawn_interval,
                spawn_jitter
            )

            gap_y = get_next_gap_y(
                previous_gap_y,
                max_gap_movement
            )

            previous_gap_y = gap_y

            pipes.append(
                Pipe(
                    WIDTH,
                    gap_y,
                    pipe_image,
                    GROUND_Y,
                    gap=gap,
                    speed=speed
                )
            )

            pipe_count += 1

        # Move pipes.
        for pipe in pipes:
            pipe.move()

        pipes = [
            pipe for pipe in pipes
            if pipe.x + pipe.width > 0
        ]

        # Move ground.
        ground_x -= 4.0

        if ground_x <= -ground.get_width():
            ground_x += ground.get_width()

        # Match training order: move bird first.
        bird.move()

        # Find next pipe.
        next_pipe = next(
            (
                pipe for pipe in pipes
                if pipe.x + pipe.width >= bird.rect.centerx
            ),
            None
        )

        # Neural network inputs.
        if next_pipe is not None:
            distance = max(
                0,
                (next_pipe.x - bird.rect.centerx) / WIDTH
            )

            gap_center = (
                next_pipe.top_rect.bottom
                + next_pipe.bottom_rect.top
            ) / 2

            inputs = (
                bird.rect.centery / HEIGHT,
                bird.velocity / 10.0,
                distance,
                gap_center / HEIGHT
            )

        else:
            inputs = (
                bird.rect.centery / HEIGHT,
                bird.velocity / 10.0,
                1.0,
                0.5
            )

        # AI decision.
        output = net.activate(inputs)[0]

        if output > JUMP_THRESHOLD:
            bird.jump()

        # Collision.
        collision = (
            bird.rect.bottom >= GROUND_Y
            or any(pipe.collide(bird) for pipe in pipes)
        )

        if collision:
            crashed = True

            print("\nAI crashed!")
            print(f"Survived: {frame_count / FPS:.1f} seconds")
            print(f"Pipes spawned: {pipe_count}")
            print(f"Pipes passed: {score}")

            break

        # Score.
        for pipe in pipes:
            if (
                pipe.x + pipe.width < bird.rect.left
                and not pipe.passed
            ):
                pipe.passed = True
                score += 1

        # Draw.
        WIN.blit(background, (0, 0))

        for pipe in pipes:
            pipe.draw(WIN)

        bird.draw(WIN)

        gx = int(ground_x)

        while gx < WIDTH:
            WIN.blit(ground, (gx, GROUND_Y))
            gx += ground.get_width()

        status = FONT.render(
            f"Score: {score}  Level: {test_level}"
            f"  Speed: {speed}",
            True,
            (255, 255, 255)
        )
        print("WIN:", type(WIN), WIN.get_size())
        print("FONT:", type(FONT))
        print("STATUS:", type(status))
        
        WIN.blit(status, (20, 20))

        pygame.display.update()

    if not crashed and running:
        if frame_count >= MAX_GENERATION_FRAMES:
            print("\n45-second test completed!")
        else:
            print("\nPlayback stopped.")

        print(f"Survived: {frame_count / FPS:.1f} seconds")
        print(f"Pipes passed: {score}")

    pygame.quit()


if __name__ == "__main__":
    main()
