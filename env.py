import pygame
import numpy as np
import random
import math
import time
from game import *


class RaindrolEnv:
    def __init__(self, render_mode=False):
        if not pygame.get_init():
            pygame.init()
        self.width, self.height = WIDTH, HEIGHT
        self.render_mode = render_mode
        if render_mode:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Raindrol RL Training")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        self.player_radius = player_radius
        self.player_speed = player_speed
        self.enemy_size = enemy_size
        self.enemy_speed = enemy_speed
        self.normal_wave_radius = normal_wave_radius
        self.charged_wave_radius = charged_wave_radius
        self.normal_wave_speed = normal_wave_speed
        self.charged_wave_speed = charged_wave_speed

        self.max_enemies = 30

        self.kill_reward = 0.0

        self.reset()

    def reset(self):
        """Reset the episode and return its first observation."""
        self.player_x = WIDTH // 2
        self.player_y = HEIGHT // 2
        self.enemies = []
        self.waves = []  # [x, y, radius, max_radius, hits, speed]
        self.score = 0
        self.charging = False
        self.charge_start_time = 0
        self.charge_canceled = False
        self.game_over = False

        self.stats_attacks = 0
        self.stats_kills = 0
        self.stats_combos = 0
        self.steps_this_episode = 0

        pygame.event.clear()

        return self._get_state()

    def _get_state(self):
        """Encode the player, ten nearest enemies, charge and score in 24 floats."""
        K = 10
        enemy_dists = []
        for ex, ey in self.enemies:
            dist = math.hypot(ex - self.player_x, ey - self.player_y)
            enemy_dists.append((dist, ex, ey))
        enemy_dists.sort(key=lambda x: x[0])

        enemy_pos = np.zeros((K, 2))
        for i, (_, ex, ey) in enumerate(enemy_dists[:K]):
            enemy_pos[i] = [ex / self.width, ey / self.height]

        charge_progress = 0.0
        if self.charging:
            hold_time = time.time() - self.charge_start_time
            charge_progress = min(hold_time / 1.0, 1.0)

        score_norm = min(self.score / 1000, 1.0)

        state = np.concatenate([
            [self.player_x / self.width, self.player_y / self.height],
            enemy_pos.flatten(),
            [charge_progress],
            [score_norm]
        ])
        return state.astype(np.float32)

    def step(self, action):
        """Apply one of 18 inputs: movement direction times two, plus the spacebar state."""
        move_action = action // 2  # 0-8
        attack_action = action % 2  # 0-1

        dx, dy = 0, 0
        if move_action == 1:
            dy = -1
        elif move_action == 2:
            dy = 1
        elif move_action == 3:
            dx = -1
        elif move_action == 4:
            dx = 1
        elif move_action == 5:
            dx, dy = -1, -1
        elif move_action == 6:
            dx, dy = 1, -1
        elif move_action == 7:
            dx, dy = -1, 1
        elif move_action == 8:
            dx, dy = 1, 1

        old_x, old_y = self.player_x, self.player_y
        new_x = self.player_x + dx * self.player_speed
        new_y = self.player_y + dy * self.player_speed
        if self.player_radius <= new_x <= self.width - self.player_radius:
            self.player_x = new_x
        if self.player_radius <= new_y <= self.height - self.player_radius:
            self.player_y = new_y

        if attack_action == 1:
            if not self.charging:
                self.charging = True
                self.charge_start_time = time.time()
        else:
            if self.charging and not self.charge_canceled:
                self.charging = False
                hold_time = time.time() - self.charge_start_time
                if hold_time >= 1.0:
                    self.waves.append([self.player_x, self.player_y, 0,
                                       self.charged_wave_radius, 0, self.charged_wave_speed])
                else:
                    self.waves.append([self.player_x, self.player_y, 0,
                                       self.normal_wave_radius, 0, self.normal_wave_speed])
                self.stats_attacks += 1
            self.charge_canceled = False

        self._update_game()

        reward = self._compute_reward()

        self.steps_this_episode += 1

        done = self.game_over

        next_state = self._get_state()

        if self.render_mode:
            self._render()

        return next_state, reward, done, {}

    def _update_game(self):
        """Advance the legacy DQN simulation by one frame."""
        for enemy in self.enemies:
            dx = self.player_x - enemy[0]
            dy = self.player_y - enemy[1]
            dist = math.hypot(dx, dy)
            if dist != 0:
                enemy[0] += self.enemy_speed * dx / dist
                enemy[1] += self.enemy_speed * dy / dist

        # The legacy trainer uses probabilistic spawns.
        if len(self.enemies) < self.max_enemies and random.random() < 0.02:
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                x, y = random.randint(0, self.width), 0
            elif side == "bottom":
                x, y = random.randint(0, self.width), self.height
            elif side == "left":
                x, y = 0, random.randint(0, self.height)
            else:
                x, y = self.width, random.randint(0, self.height)
            self.enemies.append([x, y])

        new_waves = []
        for wave in self.waves:
            wave[2] += wave[5]  # current_radius += speed
            if wave[2] < wave[3]:
                new_waves.append(wave)
            else:
                if wave[4] > 0:
                    self.score += wave[4]
                    self.stats_kills += wave[4]
                    if wave[4] > 1:
                        self.stats_combos += 1
                    if wave[5] == self.normal_wave_speed:
                        self.kill_reward += wave[4] * 10.0
                    else:
                        self.kill_reward += wave[4] * 20.0
        self.waves = new_waves

        new_enemies = []
        for enemy in self.enemies:
            hit = False
            for wave in self.waves:
                dist = math.hypot(enemy[0] - wave[0], enemy[1] - wave[1])
                if abs(dist - wave[2]) < 10:
                    wave[4] += 1
                    hit = True
                    break
            if not hit:
                new_enemies.append(enemy)
        self.enemies = new_enemies

        for enemy in self.enemies:
            dist = math.hypot(self.player_x - enemy[0], self.player_y - enemy[1])
            if dist < self.player_radius + self.enemy_size // 2:
                self.game_over = True
                break

    def _compute_reward(self):
        """Combine survival, kills and the terminal penalty."""
        reward = 0.0

        reward += 0.05

        reward += self.kill_reward
        self.kill_reward = 0.0

        if self.game_over:
            reward -= 0.5

        return reward

    def _render(self):
        """Draw the training environment."""
        if not self.render_mode or not self.screen:
            return

        self.screen.fill(GRAY)
        pygame.draw.circle(self.screen, BLACK, (int(self.player_x), int(self.player_y)), self.player_radius)
        for ex, ey in self.enemies:
            pygame.draw.rect(self.screen, RED,
                             (ex - self.enemy_size // 2, ey - self.enemy_size // 2,
                              self.enemy_size, self.enemy_size))
        for wx, wy, cr, mr, _, _ in self.waves:
            pygame.draw.circle(self.screen, BLUE, (int(wx), int(wy)), int(cr), 2)
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        if self.charging:
            charge_time = time.time() - self.charge_start_time
            charge_radius = int(charge_time * 50)
            if charge_radius >= 50:
                charge_radius = 50
            pygame.draw.circle(self.screen, CYAN, (int(self.player_x), int(self.player_y)),
                               self.player_radius + charge_radius, 2)

        pygame.display.flip()
        self.clock.tick(15)

    def close(self):
        if self.screen is not None:
            pygame.quit()
