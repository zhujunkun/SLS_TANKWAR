import torch
from parl.utils import logger, tensorboard
import random
import numpy as np
import argparse
from DQN import ReplayBuffer, DQNModel, DQN, Agent
from tank_env import TankEnv

Buffer_Size = 10000
GAMMA = 0.99
LR = 3e-4
EPSILON = 0.5
Update_Steps = 2
WARM_UP = 1000
Batch_Size = 32


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    logger.set_dir('./{}_train_{}'.format(args.env, args.seed))
    env = TankEnv('Tank Game')

    env.seed(args.seed)
    set_seed(args.seed)

    replay_buffer = ReplayBuffer(Buffer_Size)
    obs_dim = 315
    move_dim, shot_dim = 4, 2

    model = DQNModel(obs_dim=obs_dim, move_dim=move_dim, shot_dim=shot_dim)
    alg = DQN(model, gamma=GAMMA, lr=LR)
    agent = Agent(alg, epsilon=EPSILON, update_steps=Update_Steps)

    cur_steps = 0
    last_save_steps = 0
    test_flag = 0
    while cur_steps < args.total_steps:
        episode_steps, episode_rewards, epsilon = run_train_episode(replay_buffer, env, agent)
        cur_steps += episode_steps
        print(eval)
        tensorboard.add_scalar('train/episode_reward', episode_rewards, cur_steps)
        tensorboard.add_scalar('train/epsilon', epsilon, cur_steps)

        # if cur_steps > int(1e5) and cur_steps > last_save_steps + int(1e4):
        #     agent.save('./trained_mo'
        #                'del_{}/step_{}_model.ckpt'.format(args.seed, cur_steps))
        #     last_save_steps = cur_steps

        # Evaluate episode
        if (cur_steps + 1) // args.test_every_steps >= test_flag:
            while (cur_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            eval_steps, avg_reward = run_evaluate_episodes(agent, env)
            tensorboard.add_scalar('eval/episode_reward', avg_reward, cur_steps)
            logger.info(
                'Total steps {}, Evaluation over {} steps, Average reward: {}'.format(cur_steps, eval_steps, avg_reward))


def run_train_episode(replay_buffer, env, agent):
    episode_steps = 0
    episode_rewards = 0
    obs = env.reset()
    done = False
    while not done and episode_steps < 1000:
        episode_steps += 1
        action, epsilon = agent.sample(obs)
        next_obs, reward, done, info = env.step(action)
        replay_buffer.store((obs, action, reward, next_obs, done))

        episode_rewards += reward
        obs = next_obs

        if replay_buffer.size > 5*Batch_Size:
            obs_batch, act_batch, reward_batch, next_obs_batch, done_batch = replay_buffer.sample(Batch_Size)
            loss = agent.learn(obs_batch, act_batch, reward_batch, next_obs_batch, done_batch)
    logger.info("train_episode over {} steps, rewards {}".format(episode_steps, episode_rewards))
    return episode_steps, episode_rewards, epsilon


def run_evaluate_episodes(agent, env):
    avg_reward = 0
    eval_steps = 0
    obs = env.reset()
    done = False
    
    while not done and eval_steps < 1000:
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)

        avg_reward += reward
        eval_steps += 1
    avg_reward /= eval_steps
    return eval_steps, avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='TankEnv', help='game')
    parser.add_argument('--total_steps', type=int, default=int(500000), help='maximum training steps')
    parser.add_argument('--test_every_steps', type=int, default=1000, help='the step interval for evaluating')
    parser.add_argument('--seed', type=int, default=4, help='set the random seed')

    args = parser.parse_args()
    main()
