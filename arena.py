"""Headless 60 Hz training approximation of the original real-time game."""

import math
import random

WIDTH, HEIGHT, FPS = 800, 600, 60
PLAYER_RADIUS, PLAYER_SPEED = 15, 4
ENEMY_SIZE, ENEMY_SPEED = 30, 2
MOVES = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
         (-1, -1), (1, -1), (-1, 1), (1, 1))


class Arena:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.player_x, self.player_y = WIDTH // 2, HEIGHT // 2
        self.enemies = []
        self.waves = []
        self.score = self.kills = self.combos = self.attacks = 0
        self.frame = self.charge_frames = 0
        self.charge_canceled = self.game_over = False
        self.spawn_period = self.spawn_in = 90
        self.combo_until = self.combo_size = 0

    def _spawn(self):
        side = self.rng.randrange(4)
        if side < 2:
            pos = [self.rng.randint(0, WIDTH), side * HEIGHT]
        else:
            pos = [(side - 2) * WIDTH, self.rng.randint(0, HEIGHT)]
        self.enemies.append(pos)

    def clear_charge(self):
        self.charge_frames = 0
        self.charge_canceled = False

    def tick(self, action):
        """Advance exactly one frame. Return this frame's sound cues."""
        if self.game_over:
            return ()
        events = []
        self.frame += 1
        self.spawn_in -= 1
        if self.spawn_in == 0:
            self._spawn()
            self.spawn_in = self.spawn_period

        if action % 2:
            if not self.charge_canceled:
                self.charge_frames += 1
                if self.charge_frames >= 90:
                    self.charge_frames = 0
                    self.charge_canceled = True
                    events.append("cancel")
        else:
            if self.charge_frames:
                charged = self.charge_frames >= FPS
                self.waves.append([self.player_x, self.player_y, 0,
                                   150 if charged else 75, 0, 5 if charged else 3])
                self.attacks += 1
                events.append("charged" if charged else "attack")
            self.clear_charge()

        dx, dy = MOVES[action // 2]
        if (dx < 0 and self.player_x > PLAYER_RADIUS
                or dx > 0 and self.player_x < WIDTH - PLAYER_RADIUS):
            self.player_x += dx * PLAYER_SPEED
        if (dy < 0 and self.player_y > PLAYER_RADIUS
                or dy > 0 and self.player_y < HEIGHT - PLAYER_RADIUS):
            self.player_y += dy * PLAYER_SPEED
        for enemy in self.enemies:
            dx, dy = self.player_x - enemy[0], self.player_y - enemy[1]
            dist = math.hypot(dx, dy)
            if dist:
                enemy[0] += ENEMY_SPEED * dx / dist
                enemy[1] += ENEMY_SPEED * dy / dist
            if dist < PLAYER_RADIUS + ENEMY_SIZE / 2 + ENEMY_SPEED:
                self.game_over = True

        period = max(30, 90 - self.score // 10 * 6)
        if period != self.spawn_period:
            self.spawn_period = self.spawn_in = period

        active = []
        for wave in self.waves:
            wave[2] += wave[5]
            if wave[2] < wave[3]:
                active.append(wave)
            else:
                hits = wave[4]
                self.score += hits if hits < 2 else 2 * hits
                if hits > 1:
                    self.combos += 1
                    self.combo_until, self.combo_size = self.frame + FPS, hits
        self.waves = active

        survivors = []
        for enemy in self.enemies:
            for wave in self.waves:
                if abs(math.hypot(enemy[0] - wave[0], enemy[1] - wave[1]) - wave[2]) < 10:
                    wave[4] += 1
                    self.kills += 1
                    events.append("pop")
                    break
            else:
                survivors.append(enemy)
        self.enemies = survivors
        if self.game_over:
            events.append("over")
        return events
