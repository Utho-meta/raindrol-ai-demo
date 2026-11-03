"""The pilot sends the same arrows and space transitions as a human player."""

import time
from types import SimpleNamespace

import pygame

from arena import FPS, MOVES
from pilot import DECISION_FRAMES, Pilot


class AIControls:
    def __init__(self, game):
        self.game = game
        self.pilot = None
        self.enabled = False
        self.action = self.frames_left = 0
        self.space_down = False
        self.restart_at = None
        font = pygame.font.SysFont("microsoftyahei,simhei,pingfangsc,notosanscjksc", 16)
        self.labels = [font.render(text, True, (0, 0, 0)) for text in
                       ("AI Pilot: Off [F1]", "AI Pilot: On [F1]")]
        width = max(label.get_width() for label in self.labels)
        self.button = pygame.Rect(game.WIDTH - width - 20, 6, width + 14, 26)

    def events(self, events):
        game = self.game
        result = []
        for event in events:
            toggle = (event.type == pygame.KEYDOWN and event.key == pygame.K_F1
                      or event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                      and self.button.collidepoint(event.pos))
            if toggle:
                self.enabled = not self.enabled
                self.frames_left = 0
                self.restart_at = None
                if self.enabled:
                    if self.pilot is None:
                        self.pilot = Pilot.load()
                    self.space_down = game.charging or game.charge_canceled
                elif self.space_down and not pygame.key.get_pressed()[pygame.K_SPACE]:
                    result.append(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE))
                    self.space_down = False
            elif not (self.enabled and event.type in (pygame.KEYDOWN, pygame.KEYUP)
                      and event.key == pygame.K_SPACE):
                result.append(event)
        if not self.enabled:
            return result
        if game.game_over:
            now = pygame.time.get_ticks()
            if self.restart_at is None:
                self.restart_at = now + 3000
            if now >= self.restart_at:
                result.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                self.restart_at = None
            self.frames_left = self.action = 0
            return result
        self.restart_at = None
        if self.frames_left == 0:
            held = (time.time() - game.charge_start_time) * FPS if game.charging else 0
            state = SimpleNamespace(player_x=game.player_x, player_y=game.player_y,
                                    enemies=game.enemies, charge_canceled=game.charge_canceled,
                                    charge_frames=held)
            self.action = self.pilot.action(state)
            self.frames_left = DECISION_FRAMES
        self.frames_left -= 1
        hold = bool(self.action % 2)
        if hold != self.space_down:
            kind = pygame.KEYDOWN if hold else pygame.KEYUP
            result.append(pygame.event.Event(kind, key=pygame.K_SPACE))
            self.space_down = hold
        return result

    def keys(self):
        if not self.enabled:
            return pygame.key.get_pressed()
        dx, dy = MOVES[self.action // 2]
        return {pygame.K_LEFT: dx < 0, pygame.K_RIGHT: dx > 0,
                pygame.K_UP: dy < 0, pygame.K_DOWN: dy > 0}

    def draw_switch(self, screen):
        pygame.draw.rect(screen, (80, 80, 80), self.button, 1)
        label = self.labels[self.enabled]
        screen.blit(label, label.get_rect(center=self.button.center))
