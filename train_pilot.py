"""Direct policy search using the cross-entropy method and actual game scores."""

import argparse
from dataclasses import astuple, asdict
import json
from pathlib import Path
import random
from statistics import fmean, pstdev

from arena import Arena, FPS
from pilot import DECISION_FRAMES, MODEL_PATH, Pilot

# Search intervals for the eight policy coefficients, in dataclass order.
BOUNDS = ((45, 175), (0.1, 6), (40, 170), (0.05, 2.5), (0.05, 0.45),
          (150, 550), (40, 120), (110, 240))


def rollout(pilot, seed, frames):
    game = Arena(seed)
    while not game.game_over and game.frame < frames:
        if game.frame % DECISION_FRAMES == 0:
            action = pilot.action(game)
        game.tick(action)
    return game


def fitness(pilot, seeds, frames):
    results = [rollout(pilot, seed, frames) for seed in seeds]
    return fmean(g.score + 0.75 * g.kills + 0.01 * g.frame for g in results)


def train(seed, generations, population, output):
    rng = random.Random(seed)
    means = [(lo + hi) / 2 for lo, hi in BOUNDS]
    widths = [(hi - lo) / 3 for lo, hi in BOUNDS]
    champion = Pilot(*means)
    validation = range(5000, 5006)
    best = fitness(champion, validation, 180 * FPS)
    for generation in range(generations):
        seeds = rng.sample(range(1000, 5000), 4)
        pool = [champion]
        for _ in range(population - 1):
            values = [max(lo, min(hi, rng.gauss(mu, sigma)))
                      for mu, sigma, (lo, hi) in zip(means, widths, BOUNDS)]
            pool.append(Pilot(*values))
        pool.sort(key=lambda p: fitness(p, seeds, FPS * 90), reverse=True)
        elite = pool[:max(2, population // 4)]
        for i, column in enumerate(zip(*(astuple(p) for p in elite))):
            lo, hi = BOUNDS[i]
            means[i] = 0.3 * means[i] + 0.7 * fmean(column)
            widths[i] = max((hi - lo) * 0.05, 0.35 * widths[i] + 0.65 * pstdev(column))
        candidate = pool[0]
        value = fitness(candidate, validation, 180 * FPS)
        if value > best:
            best, champion = value, candidate
        print(f"Generation {generation+1:2d} | validation {value:.2f} | best {best:.2f}", flush=True)
    model = dict(version=1, algorithm="cross-entropy policy search", seed=seed,
                 generations=generations, population=population, parameters=asdict(champion))
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    if args.generations < 1 or args.population < 4:
        parser.error("need at least one generation and four candidates")
    train(args.seed, args.generations, args.population, args.output)
