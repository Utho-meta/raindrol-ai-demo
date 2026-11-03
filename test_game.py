"""Input-only integration tests for the original Raindrol loop."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pygame
from ai_controls import AIControls
from arena import FPS
from pilot import Pilot
from train_pilot import rollout


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((800, 600))

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.game = SimpleNamespace(WIDTH=800, player_x=400, player_y=300,
                                    enemies=[[455, 300]], charging=False, charge_canceled=False,
                                    charge_start_time=0, game_over=False)
        self.controls = AIControls(self.game)

    def test_manual_inputs_pass_through(self):
        press = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
        self.assertEqual(self.controls.events([press]), [press])
        keys = {pygame.K_RIGHT: True}
        with patch.object(pygame.key, "get_pressed", return_value=keys):
            self.assertIs(self.controls.keys(), keys)

    def test_click_on_f1_off_and_space_release(self):
        click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=self.controls.button.center)
        events = self.controls.events([click])
        self.assertTrue(self.controls.enabled)
        self.assertTrue(self.controls.keys()[pygame.K_LEFT])
        self.assertEqual([(e.type, e.key) for e in events], [(pygame.KEYDOWN, pygame.K_SPACE)])
        f1 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1)
        events = self.controls.events([f1])
        self.assertFalse(self.controls.enabled)
        self.assertEqual([(e.type, e.key) for e in events], [(pygame.KEYUP, pygame.K_SPACE)])

    def test_restart_uses_original_enter_action(self):
        self.game.game_over = True
        with patch.object(pygame.time, "get_ticks", side_effect=[1000, 3999, 4000]):
            self.controls.events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1)])
            self.assertFalse(self.controls.events([]))
            events = self.controls.events([])
        self.assertEqual([(e.type, e.key) for e in events], [(pygame.KEYDOWN, pygame.K_RETURN)])

    def test_held_out_simulation(self):
        game = rollout(Pilot.load(), 9103, 60 * FPS)
        self.assertFalse(game.game_over)
        self.assertGreater(game.kills, 10)


if __name__ == "__main__":
    unittest.main()
