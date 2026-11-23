from pathlib import Path
import random
import sys
import pygame
from PIL import Image

try:
    from game.bird import Bird
    from game.pipe import Pipe
except ModuleNotFoundError:
    from bird import Bird
    from pipe import Pipe

pygame.init()
pygame.font.init()

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR.parent / "assets"

# Game settings
WIDTH = 700
HEIGHT = 700
FPS = 60
GROUND_HEIGHT = 50
GROUND_Y = HEIGHT - GROUND_HEIGHT
DEBUG_HITBOXES = False

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird AI")

clock = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial", 40, bold=True)
SMALL_FONT = pygame.font.SysFont("Arial", 22, bold=True)


def load_image(path):
    image = Image.open(path).convert("RGBA")
    return pygame.image.frombytes(
        image.tobytes(), image.size, "RGBA"
    )


# Load and scale assets
BACKGROUND = load_image(ASSETS_DIR / "background.png")
BACKGROUND = pygame.transform.smoothscale(
    BACKGROUND, (WIDTH, GROUND_Y)
)

GROUND = load_image(ASSETS_DIR / "ground.png")
GROUND_WIDTH = max(
    1, int(GROUND.get_width() * GROUND_HEIGHT / GROUND.get_height())
)
GROUND = pygame.transform.smoothscale(
    GROUND, (GROUND_WIDTH, GROUND_HEIGHT)
)

PIPE_IMAGE = load_image(ASSETS_DIR / "pipe.png")


def main():
    running = True
    ground_x = 0.0

    bird = Bird(100, 300)
    pipes = []

    score = 0
    game_active = True

    SPAWN_PIPE = pygame.USEREVENT + 1
    pygame.time.set_timer(SPAWN_PIPE, 1800)

    def reset_game():
        nonlocal bird, pipes, score, game_active
        bird = Bird(100, 300)
        pipes = []
        score = 0
        game_active = True

    while running:
        clock.tick(FPS)

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if game_active:
                        bird.jump()
                    else:
                        reset_game()
                elif event.key == pygame.K_r:
                    reset_game()

            if event.type == SPAWN_PIPE and game_active:
                min_gap_y = 220
                max_gap_y = max(min_gap_y + 1, GROUND_Y - 120)
                gap_y = random.randint(min_gap_y, max_gap_y)
                pipes.append(
                    Pipe(WIDTH, gap_y, PIPE_IMAGE, GROUND_Y)
                )

        # Draw background
        WIN.blit(BACKGROUND, (0, 0))

        # Game logic
        if game_active:
            bird.move()

            # Scroll ground continuously
            ground_x -= 4.0
            if ground_x <= -GROUND_WIDTH:
                ground_x += GROUND_WIDTH

            # Ground collision
            if bird.rect.bottom >= GROUND_Y:
                bird.rect.bottom = GROUND_Y
                bird.y = float(bird.rect.y - bird.hitbox_offset_y)
                game_active = False

            # Pipe movement, collision, and scoring
            for pipe in pipes:
                pipe.move()

                if game_active and pipe.collide(bird):
                    game_active = False

                pipe_center_x = pipe.x + pipe.width / 2
                if not pipe.passed and pipe_center_x < bird.rect.centerx:
                    pipe.passed = True
                    score += 1

            # Remove off-screen pipes
            pipes = [
                pipe for pipe in pipes
                if pipe.x + pipe.width > 0
            ]

        # Draw pipes
        for pipe in pipes:
            pipe.draw(WIN, debug=DEBUG_HITBOXES)

        # Draw bird
        bird.draw(WIN, debug=DEBUG_HITBOXES)

        # Draw ground (tiled across screen width)
        gx = int(ground_x)
        while gx < WIDTH:
            WIN.blit(GROUND, (gx, GROUND_Y))
            gx += GROUND_WIDTH

        # Display score
        score_text = FONT.render(
            f"Score: {score}", True, (255, 255, 255)
        )
        WIN.blit(score_text, (20, 20))

        # Game over screen
        if not game_active:
            game_over_text = FONT.render(
                "GAME OVER", True, (255, 80, 80)
            )
            text_rect = game_over_text.get_rect(
                center=(WIDTH // 2, HEIGHT // 2 - 20)
            )
            WIN.blit(game_over_text, text_rect)

            restart_text = SMALL_FONT.render(
                "Press SPACE or R to restart", True, (255, 255, 255)
            )
            restart_rect = restart_text.get_rect(
                center=(WIDTH // 2, HEIGHT // 2 + 25)
            )
            WIN.blit(restart_text, restart_rect)

        pygame.display.update()

    pygame.time.set_timer(SPAWN_PIPE, 0)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()