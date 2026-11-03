"""Compact policy for the game's 18 legal inputs. Only uses visible positions."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from arena import HEIGHT, WIDTH, MOVES

DECISION_FRAMES = 4
BUTTONS = {move: i * 2 for i, move in enumerate(MOVES)}
MODEL_PATH = Path(__file__).with_name("pilot.json")


@dataclass(frozen=True)
class Pilot:
    spacing: float
    center_pull: float
    crowd_radius: float
    crowd_push: float
    dead_zone: float
    start_charge: float
    release_normal: float
    release_charged: float

    @classmethod
    def load(cls, path=MODEL_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["version"] != 1:
            raise ValueError("Unsupported pilot version")
        return cls(**data["parameters"])

    def action(self, game):
        x, y = game.player_x, game.player_y
        steer_x = self.center_pull * ((WIDTH / 2 - x) / (WIDTH / 2)) ** 3
        steer_y = self.center_pull * ((HEIGHT / 2 - y) / (HEIGHT / 2)) ** 3
        nearest = math.inf
        threat_x = threat_y = 0
        for ex, ey in game.enemies:
            dx, dy = ex - x, ey - y
            dist = max(0.01, math.hypot(dx, dy))
            if dist < nearest:
                nearest, threat_x, threat_y = dist, dx / dist, dy / dist
            if dist < self.crowd_radius:
                push = self.crowd_push * (1 - dist / self.crowd_radius)
                steer_x -= push * dx / dist
                steer_y -= push * dy / dist
        approach = max(-1, min(1, (nearest - self.spacing) / 60))
        steer_x += approach * threat_x
        steer_y += approach * threat_y
        dx = int(steer_x > self.dead_zone) - int(steer_x < -self.dead_zone)
        dy = int(steer_y > self.dead_zone) - int(steer_y < -self.dead_zone)

        charge = game.charge_frames
        if game.charge_canceled:
            hold = False
        elif not charge:
            hold = nearest <= self.start_charge
        elif charge >= 60:
            hold = nearest > self.release_charged and charge < 84
        else:
            hold = nearest > self.release_normal
        return BUTTONS[dx, dy] + int(hold)
