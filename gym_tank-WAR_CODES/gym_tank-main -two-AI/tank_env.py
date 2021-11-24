import gym
import numpy as np
from sprites import *
from env_config import *
from parl.utils import logger


class TankEnv(gym.Env):
    def __init__(self, env_name):
        pygame.init()
        pygame.display.set_caption(env_name)
        self.screen = pygame.display.set_mode(SCREEN_RECT.size)
        self.clock = pygame.time.Clock()
        self.game_alive = True
        self.time_step = 0
        self._max_time_episode = MAX_TIME_EPISODE


    def __init_game(self):
        self.hero = Hero(screen=self.screen)
        self.enemy = Enemy(screen=self.screen)
        self.walls = pygame.sprite.Group()

        self.__init_map()

    def reset(self):
        self.__init_game()
        state = self._get_obs()
        self.__update_map()
        return state

    def step(self, action, action_enemy):
        for enent in pygame.event.get():
            if enent.type == pygame.QUIT:
                self.close()


        if self.hero.is_alive and self.game_alive:
            self.screen.fill((0, 0, 0))
            self.clock.tick(FPS)

            move, shot = action[0], action[1]
            self.hero.direction = move
            self.hero.is_hit_wall = False
            if shot:
                self.hero.shot()
            # hero_reward = self.action_check_out()

        if self.enemy.is_alive and self.game_alive:
            self.screen.fill((0, 0, 0))


            move, shot = action_enemy[0], action_enemy[1]
            self.enemy.direction = move
            self.enemy.is_hit_wall = False
            if shot:
                self.enemy.shot()
            # hero_reward = self.action_check_out()

            # update and draw the world
            hero_reward,enemy_reward = self.__update_map()
            reward_enemy=enemy_reward
            reward = hero_reward
            is_done = self._terminal()
            info = None
        else:
            is_done = True
            info="game over"
        state = self._get_obs()
        return state, hero_reward, enemy_reward, is_done, info
    
    
    
    def _terminal(self):
        if self.time_step >= self._max_time_episode:
            self.time_out = True
            return True
        if not self.game_alive:
            return True
        return False

    def action_check_out(self):
        self.hero.hit_wall()
        self.enemy.hit_wall()

        hero_reward = 0
        enemy_reward = 0

        for bullet in self.hero.bullets:
            if pygame.sprite.collide_rect(bullet, self.enemy):
                bullet.kill()
                self.enemy.kill()
                hero_reward += 100000
                enemy_reward-=10000
                self.game_alive = False
        for bullet in self.enemy.bullets:
            if pygame.sprite.collide_rect(bullet, self.hero):
                bullet.kill()
                self.hero.kill()
                #敌人打死自己
                hero_reward -= 10000
                enemy_reward+=10000
                self.game_alive = False

        if pygame.sprite.collide_rect(self.enemy, self.hero):
                # 不可穿越墙
            self.enemy.kill()
            self.hero.kill()
            hero_reward -= 10000
            enemy_reward -= 10000
            self.game_alive = False
        for wall in self.walls:
            for bullet in self.hero.bullets:
                if pygame.sprite.collide_rect(wall, bullet):
                    if wall.type == RED_WALL:
                        bullet.kill()
                        localtion = wall.kill()
                        self.map_state[localtion[0]][localtion[1]] = 0
                        # hero_reward += 1
                        # enemy_reward+= 1
                    elif wall.type == BOSS_WALL:
                        #打死了自己家-100
                        self.game_alive = False
                        hero_reward -= 100000
                        enemy_reward+=10000
                        

                    elif wall.type == IRON_WALL:
                        bullet.kill()
            # Hero hit the wall
            if pygame.sprite.collide_rect(self.hero, wall):
                # 不可穿越墙
                if wall.type == RED_WALL or wall.type == IRON_WALL or wall.type == BOSS_WALL:
                    self.hero.is_hit_wall = True
                    # 移出墙内
                    self.hero.out_of_wall(wall)
                    hero_reward-=20
                    
            if pygame.sprite.collide_rect(self.enemy, wall):
                # 不可穿越墙
                if wall.type == RED_WALL or wall.type == IRON_WALL or wall.type == BOSS_WALL:
                    self.enemy.is_hit_wall = True
                    # 移出墙内
                    self.enemy.out_of_wall(wall)
                    enemy_reward-=20
            for bullet in self.enemy.bullets:
                if pygame.sprite.collide_rect(wall, bullet):
                    if wall.type == RED_WALL:
                        bullet.kill()
                        localtion = wall.kill()
                        self.map_state[localtion[0]][localtion[1]] = 0
                        enemy_reward+=1
                    elif wall.type == BOSS_WALL:
                        self.game_alive = False
                        hero_reward -= 5000
                        enemy_reward += 5000
                    elif wall.type == IRON_WALL:
                        bullet.kill()

            hero_reward-=1
            enemy_reward-=1
        return hero_reward, enemy_reward

    def render(self, display=True):
        if not display:
            os.environ['SDL_VIDEODRIVER'] = "dummy"

    def close(self):
        pygame.quit()
        exit()

    def _get_obs(self):
        map_state = self.map_state.reshape(-1)
        hero_state = [self.hero.rect.centerx, self.hero.rect.bottom]
        enemy_state = [self.enemy.rect.centerx, self.enemy.rect.bottom]
        state = np.concatenate([map_state])
        return state

    def _get_reward(self):
        pass

    def __update_map(self):
        self.hero.update()
        self.hero.bullets.update()
        self.walls.update()
        self.enemy.update()
        self.enemy.bullets.update()
        hero_reward, enemy_reward = self.action_check_out()
        self.hero.bullets.draw(self.screen)
        self.enemy.bullets.draw(self.screen)
        self.walls.draw(self.screen)
        self.screen.blit(self.hero.image, self.hero.rect)
        self.screen.blit(self.enemy.image, self.enemy.rect)
        pygame.display.update()
        return hero_reward,enemy_reward
    

    
    def __init_map(self):
        self.map_state = MAP
        for x in range(len(self.map_state)):
            for y in range(len(self.map_state[x])):
                if self.map_state[x][y] == 0:
                    continue
                wall = Wall(wall_type=self.map_state[x][y], x_y=[x, y], screen=self.screen)
                self.walls.add(wall)
        self.map_state = np.array(self.map_state)

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
