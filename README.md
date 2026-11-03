# Raindrol

Dodge the red squares and clear them with expanding ripples. Play by hand or let the trained AI pilot take over.

[Download for Windows](https://github.com/Utho-meta/raindrol-ai-demo/releases/latest/download/Raindrol.exe)

| Input | Action |
| --- | --- |
| Arrow keys | Move |
| Space | Tap to attack. Hold for 1 second and release for a larger ripple. |
| F1 or the AI Pilot button | Toggle the pilot mid-game |
| Enter | Restart after a loss |

## Run from source

Run these commands from the repository folder.

```sh
python -m pip install pygame==2.6.1
python game.py
```

## Train and evaluate

The pilot uses cross-entropy search to tune movement and attack timing. It survived all 30 held-out five-minute simulations, averaging 490 kills per run.

```sh
python train_pilot.py
python evaluate.py
```

To compare the original DQN checkpoints, run `python -m pip install -r requirements.txt` and run `python evaluate.py --compare-dqn`.
