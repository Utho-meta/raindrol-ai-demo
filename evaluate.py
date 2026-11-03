"""Paired games on held-out spawn seeds. Never restart after a loss."""

import argparse
from pathlib import Path
from statistics import fmean

from arena import Arena, FPS
from pilot import Pilot
from train_pilot import rollout


def legacy_rollouts(checkpoint, seeds, frames):
    import torch
    from train import QNetwork
    torch.set_num_threads(1)
    network = QNetwork(24, 18)
    network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    network.eval()
    with torch.inference_mode():
        for seed in seeds:
            game = Arena(seed)
            while game.frame < frames and not game.game_over:
                x, y = game.player_x, game.player_y
                enemies = sorted(game.enemies, key=lambda e: (e[0]-x)**2 + (e[1]-y)**2)
                state = torch.zeros(24)
                state[:2] = torch.tensor([x / 800, y / 600])
                for i, (ex, ey) in enumerate(enemies[:10]):
                    state[2+2*i:4+2*i] = torch.tensor([ex / 800, ey / 600])
                state[-2] = min(1, game.charge_frames / FPS)
                state[-1] = min(1, game.score / 1000)
                game.tick(network(state).argmax().item())
            yield game


def report(label, games):
    results = list(games)
    alive = sum(not game.game_over for game in results)
    print(f"{label:17}  {alive:3}/{len(results):<3}  "
          f"{fmean(g.frame for g in results)/FPS:8.2f}  "
          f"{fmean(g.kills for g in results):9.2f}  {fmean(g.score for g in results):10.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=8000)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--seconds", type=int, default=300)
    p.add_argument("--compare-dqn", action="store_true")
    args = p.parse_args()
    if args.games < 1 or args.seconds < 1:
        p.error("need positive game count and horizon")
    seeds = range(args.seed, args.seed + args.games)
    frames = FPS * args.seconds
    print("policy             survived   seconds      kills       score")
    pilot = Pilot.load()
    report("autopilot", (rollout(pilot, seed, frames) for seed in seeds))
    if args.compare_dqn:
        for name in ("dqn_model.pth", "dqn_final.pth"):
            report(name, legacy_rollouts(Path(__file__).with_name(name), seeds, frames))
