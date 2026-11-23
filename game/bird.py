from pathlib import Path
import pygame
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR.parent / "assets"


def load_image(path):
    image = Image.open(path).convert("RGBA")
    return pygame.image.frombytes(
        image.tobytes(), image.size, "RGBA"
    )


class Bird:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

        self.width = 50
        self.height = 50

        self.velocity = 0.0
        self.gravity = 0.5
        self.jump_strength = -8.0

        self.image = load_image(ASSETS_DIR / "bird.png")
        self.image = pygame.transform.smoothscale(
            self.image, (self.width, self.height)
        )

        # Inset hitbox (34x34) to account for surrounding sprite padding
        self.hitbox_offset_x = 8
        self.hitbox_offset_y = 8
        self.rect = pygame.Rect(
            int(self.x) + self.hitbox_offset_x,
            int(self.y) + self.hitbox_offset_y,
            self.width - 2 * self.hitbox_offset_x,
            self.height - 2 * self.hitbox_offset_y,
        )

    def sync_rect(self):
        self.rect.x = int(self.x) + self.hitbox_offset_x
        self.rect.y = int(self.y) + self.hitbox_offset_y

    def jump(self):
        self.velocity = self.jump_strength

    def move(self):
        self.velocity += self.gravity
        self.y += self.velocity

        # Clamp at top of screen instead of triggering instant Game Over
        if self.y < -self.hitbox_offset_y:
            self.y = float(-self.hitbox_offset_y)
            if self.velocity < 0:
                self.velocity = 0.0

        self.sync_rect()

    def draw(self, window, debug=False):
        window.blit(self.image, (int(self.x), int(self.y)))
        if debug:
            pygame.draw.rect(window, (255, 0, 0), self.rect, 2)