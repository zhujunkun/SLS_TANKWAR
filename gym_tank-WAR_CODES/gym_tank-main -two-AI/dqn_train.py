import torch
from parl.utils import logger, tensorboard
import random
import numpy as np
import argparse
from DQN import ReplayBuffer, DQNModel, DQN, Agent,DQNModel_enemy,Agent_enemy
from tank_env import TankEnv
import os
Buffer_Size = 10000
GAMMA = 0.98
LR = 0.005
EPSILON = 0.4
Batch_Size = 32
Update_Steps=2

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def run_train_episode(replay_buffer,replay_buffer_enemy, env, agent,agent_enemy):
    episode_steps = 0
    episode_rewards = 0
    episode_rewards_enemy = 0
    obs = env.reset()
    done = False
    while not done and episode_steps < args.max_steps_one:
        episode_steps += 1
        action, epsilon = agent.sample(obs)
        action_enemy, epsilon_enemy = agent_enemy.sample(obs)
        next_obs, reward, reward_enemy, done, info = env.step(action,action_enemy)
        if(done==True):
            env.__init__("TankEnv")
        replay_buffer.store((obs, action, reward, next_obs, done))
        replay_buffer_enemy.store((obs, action_enemy, reward_enemy, next_obs, done))

        episode_rewards += reward
        episode_rewards_enemy +=reward_enemy
        obs = next_obs
        if replay_buffer.size > 5*Batch_Size:
            obs_batch, act_batch, reward_batch, next_obs_batch, done_batch = replay_buffer.sample(Batch_Size)
            loss = agent.learn(obs_batch, act_batch, reward_batch, next_obs_batch, done_batch)
            obs_batch_enemy, act_batch_enemy, reward_batch_enemy, next_obs_batch_enemy, done_batch_enemy = replay_buffer_enemy.sample(Batch_Size)
            loss_enemy = agent_enemy.learn(obs_batch_enemy, act_batch_enemy, reward_batch_enemy, next_obs_batch_enemy, done_batch_enemy)
    if(episode_rewards!=0 or  episode_rewards_enemy!=0):      
        logger.info("train_episode over {} steps, rewards {}".format(episode_steps, episode_rewards))
        logger.info("train_episode over {} steps, rewards {} for enemy".format(episode_steps, episode_rewards_enemy))
    return episode_steps, episode_rewards, epsilon,episode_rewards_enemy, epsilon_enemy




def run_evaluate_episodes(agent,agent_enemy, env):
    avg_reward = 0
    avg_reward_enemy=0
    eval_steps = 0
    obs = env.reset()
    done = False
    
    while not done and eval_steps < env._max_time_episode:
        action = agent.predict(obs)
        action_enemy = agent_enemy.predict(obs)
        obs, reward, reward_enemy, done, info = env.step(action, action_enemy)
        if(done==True):
            env.__init__("TankEnv")
        avg_reward += reward
        avg_reward_enemy+=reward_enemy
        eval_steps += 1
    avg_reward /= eval_steps
    avg_reward_enemy/=eval_steps
    return eval_steps, avg_reward,avg_reward_enemy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=int(50000), help='maximum training steps')
    parser.add_argument('--test_every_steps', type=int, default=200, help='the step interval for evaluating')
    parser.add_argument('--max_steps_one', type=int, default=200, help='the step interval for evaluating')
    parser.add_argument('--model_hero_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--model_enemy_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--seed', type=int, default=4, help='set the random seed')
    args = parser.parse_args()

    env = TankEnv('Tank Game')

    env.seed(args.seed)
    set_seed(args.seed)

    replay_buffer = ReplayBuffer(Buffer_Size)
    replay_buffer_enemy = ReplayBuffer(Buffer_Size)
    obs_dim = 315
    move_dim, shot_dim = 4, 2

    model_enemy = DQNModel_enemy(obs_dim=obs_dim, move_dim=move_dim, shot_dim=shot_dim)
        
    if args.model_enemy_path:
        model_enemy.load_state_dict(torch.load(args.model_enemy_path))
        print('Model loaded : {}'.format(args.model_enemy_path))

    alg_enemy = DQN(model_enemy, gamma=GAMMA, lr=LR)
    agent_enemy = Agent_enemy(alg_enemy, epsilon=EPSILON, update_steps=Update_Steps)
    
    model = DQNModel(obs_dim=obs_dim, move_dim=move_dim, shot_dim=shot_dim)
    if args.model_hero_path:
        model.load_state_dict(torch.load(args.model_hero_path))
        print('Model loaded : {}'.format(args.model_hero_path))
    
    alg = DQN(model, gamma=GAMMA, lr=LR)
    agent = Agent(alg, epsilon=EPSILON, update_steps=Update_Steps)

    cur_steps = 0
    last_save_steps = 0
    test_flag = 0
    while cur_steps < args.total_steps:
        episode_steps, episode_rewards, epsilon,episode_rewards_enemy, epsilon_enemy = run_train_episode(replay_buffer,replay_buffer_enemy, env, agent, agent_enemy)
        cur_steps += episode_steps
        print(eval)
        tensorboard.add_scalar('train/episode_reward', episode_rewards, cur_steps)
        tensorboard.add_scalar('train/epsilon', epsilon, cur_steps)
        tensorboard.add_scalar('train/episode_reward_enemy', episode_rewards_enemy, cur_steps)
        tensorboard.add_scalar('train/epsilon_enemy', epsilon_enemy, cur_steps)

        # if cur_steps > int(1e5) and cur_steps > last_save_steps + int(1e4):
        #     agent.save('./trained_mo'
        #                'del_{}/step_{}_model.ckpt'.format(args.seed, cur_steps))
        #     last_save_steps = cur_steps

        # Evaluate episode
        if (cur_steps + 1) // args.test_every_steps >= test_flag:
            while (cur_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            eval_steps, avg_reward,avg_reward_enemy = run_evaluate_episodes(agent,agent_enemy, env)
            logger.info('Total steps {}, Evaluation over {} steps, Average reward: {}'.format(cur_steps, eval_steps, avg_reward))
            logger.info('Total steps {}, Evaluation over {} steps, Average reward: {} for enemy'.format(cur_steps, eval_steps, avg_reward_enemy))

    torch.save(model.state_dict(), os.path.join('hero.pth'))
    torch.save(model_enemy.state_dict(), os.path.join('enemy.pth'))


