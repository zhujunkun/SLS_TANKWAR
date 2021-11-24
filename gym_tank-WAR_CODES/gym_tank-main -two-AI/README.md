#How to use our project
DQN.py is for DQN algorithms and models
env_config.py is for map design and some parameters of our env
tank_env.py is the main file for the tank war environment
To run our system
If you run the system for the first time   python dqn_train.py --total_steps=10000 --test_every_steps=1000 --max_steps_one=200
The model will be saved as hero.pth and enemy.pth
If you have saved a model  python dqn_train.py --total_steps=10000 --test_every_steps=1000 --max_steps_one=200 --model_hero_path=hero.pth --model_enemy_path=enemy.pth



# gym_tank
RL environment of Tank War.  
A tank game wrapper of gym with pygame.


#### state
A one dimension vector of [map_info]

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