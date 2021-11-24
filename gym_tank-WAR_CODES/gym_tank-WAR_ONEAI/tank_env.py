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
        self.enemies = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        for i in range(Enemy_Num):
            enemy = Enemy(screen=self.screen)
            self.enemies.add(enemy)
        self.__init_map()

    def reset(self):
        self.__init_game()
        state = self._get_obs()
        self.__update_map()
        return state

    def step(self, action):
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

            # update and draw the world
            hero_reward = self.__update_map()

            reward = hero_reward
            is_done = self._terminal()
            info = None
        else:
            is_done = True
            self.__init__("TankEnv")
            reward=0
            info="game over"
        state = self._get_obs()
        return state, reward, is_done, info

    def _terminal(self):
        if self.time_step >= self._max_time_episode:
            self.time_out = True
            return True
        if not self.game_alive:
            return True
        return False

    def action_check_out(self):
        self.hero.hit_wall()
        for enemy in self.enemies:
            enemy.hit_wall()

        hero_reward = 0
        for enemy in self.enemies:
            for bullet in self.hero.bullets:
                if pygame.sprite.collide_rect(bullet, enemy):
                    bullet.kill()
                    enemy.kill()
                    hero_reward += 10
        for enemy in self.enemies:
            for bullet in enemy.bullets:
                if pygame.sprite.collide_rect(bullet, self.hero):
                    bullet.kill()
                    self.hero.kill()
                    #敌人打死自己
                    hero_reward -= 100

        for wall in self.walls:
            for bullet in self.hero.bullets:
                if pygame.sprite.collide_rect(wall, bullet):
                    if wall.type == RED_WALL:
                        bullet.kill()
                        localtion = wall.kill()
                        self.map_state[localtion[0]][localtion[1]] = 0
                        # hero_reward += 1
                    elif wall.type == BOSS_WALL:
                        #打死了自己家-100
                        self.game_alive = False
                        hero_reward -= 100

                    elif wall.type == IRON_WALL:
                        bullet.kill()
            # Hero hit the wall
            if pygame.sprite.collide_rect(self.hero, wall):
                # 不可穿越墙
                if wall.type == RED_WALL or wall.type == IRON_WALL or wall.type == BOSS_WALL:
                    self.hero.is_hit_wall = True
                    # 移出墙内
                    self.hero.out_of_wall(wall)
            for enemy in self.enemies:
                for bullet in enemy.bullets:
                    if pygame.sprite.collide_rect(wall, bullet):
                        if wall.type == RED_WALL:
                            bullet.kill()
                            localtion = wall.kill()
                            self.map_state[localtion[0]][localtion[1]] = 0
                            hero_reward -= 0
                        elif wall.type == BOSS_WALL:
                            self.game_alive = False
                            hero_reward -= 100
                        elif wall.type == IRON_WALL:
                            bullet.kill()
                if pygame.sprite.collide_rect(wall, enemy):
                    if wall.type == RED_WALL or wall.type == IRON_WALL or wall.type == BOSS_WALL:
                        enemy.out_of_wall(wall)
                        enemy.random_turn()
        return hero_reward

    def render(self, display=True):
        if not display:
            os.environ['SDL_VIDEODRIVER'] = "dummy"

    def close(self):
        pygame.quit()
        exit()

    def _get_obs(self):
        map_state = self.map_state.reshape(-1)
        hero_state = [self.hero.rect.centerx, self.hero.rect.bottom]
        enemies_state = []
        for enemy in self.enemies:
            enemy_state = [enemy.rect.centerx, enemy.rect.bottom]
            enemies_state += enemy_state
        state = np.concatenate([map_state])
        return state

    def _get_reward(self):
        pass

    def __update_map(self):
        self.hero.update()
        self.hero.bullets.update()
        self.walls.update()
        self.enemies.update()
        for enemy in self.enemies:
            enemy.bullets.update()
            enemy.bullets.draw(self.screen)
        hero_reward = self.action_check_out()
        self.hero.bullets.draw(self.screen)
        self.enemies.draw(self.screen)
        self.walls.draw(self.screen)
        self.screen.blit(self.hero.image, self.hero.rect)
        pygame.display.update()
        return hero_reward

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
