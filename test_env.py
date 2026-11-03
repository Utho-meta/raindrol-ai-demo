import traceback
import pygame
import sys

try:
    from env import RaindrolEnv
    import random
    import time

    print("=== test_env.py started ===")
    env = RaindrolEnv(render_mode=True)
    print("after env creation")
    state = env.reset()
    print("State shape:", state.shape)

    for step in range(1000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                sys.exit()

        action = random.randint(0, 17)
        next_state, reward, done, _ = env.step(action)
        print(f"Step {step}, Action: {action}, Reward: {reward:.3f}, Done: {done}")
        if done:
            print("Game over, resetting...")
            state = env.reset()
        else:
            state = next_state
        time.sleep(0.1)

except Exception as e:
    print("Exception occurred:")
    traceback.print_exc()
