# gym_tank
RL environment of Tank War.  
A tank game wrapper of gym with pygame.

## Implement example
There is an example in `example.py`. Just the same as `gym` wrapper.

```python
import random
from tank_env import TankEnv

env = TankEnv('Tank_game')
env.seed(0)
state = env.reset()
done = False

episode_reward = 0
episode_steps = 0
while not done and episode_steps <= env._max_time_episode:
    action = [random.randint(0, 3), random.randint(0, 1)]
    next_state, reward, done, _ = env.step(action)
    
    state = next_state
    episode_reward += reward
    episode_steps += 1
```
#### state
A one dimension vector of [map_info, hero_location, enemies_location]

#### action
[move, shot]  
move: range from (0, 1, 2, 3) represents (left, right, up, down) respectively.  
shot: range from (0, 1) represents (not shot, shot) respectively.

#### reward function
positive reward:   
hero kill an enemy (+10)  


negative reward:  
hero be killed by enemy: (-100)  
home be shot by enemy: (-100)  
hero shot home: (-100)
### reference
TankWar from https://github.com/IronSpiderMan/TankWar.