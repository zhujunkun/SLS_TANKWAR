from tank_env import TankEnv
from parl.utils import logger
import random

if __name__ == "__main__":
    env = TankEnv('Tank_game')
    env.render(display=False)
    env.seed(0)
    state = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0

    logger.info('rewards: {}, steps: {}'.format(episode_reward, episode_steps))
    while not done and episode_steps <= 10000:
        episode_steps += 1
        action = [2, 1]
        state, reward, done, _ = env.step(action)
        episode_reward += reward
    logger.info('rewards: {}, steps: {}'.format(episode_reward, episode_steps))
    state = env.reset()
