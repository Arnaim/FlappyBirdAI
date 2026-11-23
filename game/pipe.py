
import pygame


class Pipe:
    def __init__(
        self,
        x,
        gap_y,
        pipe_image,
        ground_y,
        gap=180,
        speed=4.0
    ):
        self.x = float(x)
        self.width = 80
        self.gap = gap
        self.speed = speed
        self.ground_y = ground_y

        # Calculate the gap boundaries
        gap_top = gap_y - self.gap // 2
        gap_bottom = gap_y + self.gap // 2

        # Ensure positive heights for smoothscale
        top_height = max(1, gap_top)
        bottom_height = max(1, ground_y - gap_bottom)

        # Resize pipe images
        self.top_image = pygame.transform.smoothscale(
            pipe_image, (self.width, top_height)
        )
        self.bottom_image = pygame.transform.smoothscale(
            pipe_image, (self.width, bottom_height)
        )

        # Flip the upper pipe so its rim faces down toward the gap
        self.top_image = pygame.transform.flip(
            self.top_image, False, True
        )

        # Position pipes
        self.top_rect = self.top_image.get_rect(
            topleft=(int(self.x), 0)
        )
        self.bottom_rect = self.bottom_image.get_rect(
            topleft=(int(self.x), gap_bottom)
        )

        # Compute exact non-transparent bounds from scaled/flipped pipe surfaces
        top_mask_rects = pygame.mask.from_surface(
            self.top_image
        ).get_bounding_rects()

        if top_mask_rects:
            top_bounds = top_mask_rects[0].unionall(top_mask_rects[1:])
            self._top_hitbox_offset_x = top_bounds.x + 6
            self.top_hitbox = pygame.Rect(
                self.top_rect.x + self._top_hitbox_offset_x,
                self.top_rect.y + top_bounds.y,
                max(1, top_bounds.width - 12),
                max(1, top_bounds.height),
            )
        else:
            self._top_hitbox_offset_x = 0
            self.top_hitbox = self.top_rect.copy()

        bottom_mask_rects = pygame.mask.from_surface(
            self.bottom_image
        ).get_bounding_rects()

        if bottom_mask_rects:
            bottom_bounds = bottom_mask_rects[0].unionall(
                bottom_mask_rects[1:]
            )
            self._bottom_hitbox_offset_x = bottom_bounds.x + 6
            self.bottom_hitbox = pygame.Rect(
                self.bottom_rect.x + self._bottom_hitbox_offset_x,
                self.bottom_rect.y + bottom_bounds.y,
                max(1, bottom_bounds.width - 12),
                max(1, bottom_bounds.height),
            )
        else:
            self._bottom_hitbox_offset_x = 0
            self.bottom_hitbox = self.bottom_rect.copy()

        self.passed = False

    def move(self):
        self.x -= self.speed

        px = int(self.x)
        self.top_rect.x = px
        self.bottom_rect.x = px

        self.top_hitbox.x = px + self._top_hitbox_offset_x
        self.bottom_hitbox.x = px + self._bottom_hitbox_offset_x

    def draw(self, window, debug=False):
        window.blit(self.top_image, self.top_rect)
        window.blit(self.bottom_image, self.bottom_rect)

        if debug:
            pygame.draw.rect(window, (0, 255, 0), self.top_hitbox, 2)
            pygame.draw.rect(window, (0, 255, 0), self.bottom_hitbox, 2)

    def collide(self, bird):
        return (
            bird.rect.colliderect(self.top_hitbox)
            or bird.rect.colliderect(self.bottom_hitbox)
        )
