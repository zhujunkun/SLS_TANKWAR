import os
import random
import time
from threading import Thread
import pygame
from env_config import *


class Hero(pygame.sprite.Sprite):
    def __init__(self, screen):
        super(Hero, self).__init__()
        self.screen = screen
        self.image = pygame.image.load('.images/hero/hero3U.gif')
        self.rect = self.image.get_rect()
        self.type = Hero_Type
        self.direction = UP
        self.speed = Hero_Speed
        self.is_alive = True
        self.is_hit_wall = False
        self.bullets = pygame.sprite.Group()

        # init localtion
        self.rect.centerx = SCREEN_RECT.centerx - BOX_RECT.width * 3
        self.rect.bottom = SCREEN_RECT.bottom  - BOX_RECT.width

    def _get_state(self):
        pass

    def shot(self):
        self.__remove_bullets()
        if not self.is_alive or len(self.bullets) >= 3:
            return
        bullet = Bullet(screen=self.screen)
        bullet.direction = self.direction
        if self.direction == LEFT:
            bullet.rect.right = self.rect.left
            bullet.rect.centery = self.rect.centery
        elif self.direction == RIGHT:
            bullet.rect.left = self.rect.right
            bullet.rect.centery = self.rect.centery
        elif self.direction == UP:
            bullet.rect.bottom = self.rect.top
            bullet.rect.centerx = self.rect.centerx
        elif self.direction == DOWN:
            bullet.rect.top = self.rect.bottom
            bullet.rect.centerx = self.rect.centerx
        self.bullets.add(bullet)

    def update(self):
        if self.direction == LEFT:
            self.rect.x -= self.speed
        elif self.direction == RIGHT:
            self.rect.x += self.speed
        elif self.direction == UP:
            self.rect.y -= self.speed
        elif self.direction == DOWN:
            self.rect.y += self.speed

        if not self.is_hit_wall:
            self.image = pygame.image.load([f'.images/hero/{file}' for file in os.listdir('.images/hero/')][self.direction])
        if not self.is_alive:
            return

    def __remove_bullets(self):
        for bullet in self.bullets:
            rect = bullet.rect
            if rect.bottom <= 0 or rect.top >= SCREEN_RECT.bottom or rect.right <= 0 or rect.left >= SCREEN_RECT.right:
                self.bullets.remove(bullet)
                bullet.kill()

    def hit_wall(self):
        if (self.direction == LEFT and self.rect.left <= 0) \
                or (self.direction == RIGHT and self.rect.right >= SCREEN_RECT.right) \
                or (self.direction == UP and self.rect.top <= 0) \
                or (self.direction == DOWN and self.rect.bottom >= SCREEN_RECT.bottom):
            self.is_hit_wall = True

    def out_of_wall(self, wall):
        if self.direction == LEFT:
            self.rect.left = wall.rect.right + 2
        elif self.direction == RIGHT:
            self.rect.right = wall.rect.left - 2
        elif self.direction == UP:
            self.rect.top = wall.rect.bottom + 2
        elif self.direction == DOWN:
            self.rect.bottom = wall.rect.top - 2

    def boom(self):
        for sub_boom in ['.images/boom/' + file for file in os.listdir('.images/boom')]:
            self.image = pygame.image.load(sub_boom)
            time.sleep(Tank_Boom_Sleep)
            self.screen.blit(self.image, self.image.get_rect())
        super().kill()

    def kill(self):
        self.is_alive = False
        t = Thread(target=self.boom)
        t.start()


class Enemy(pygame.sprite.Sprite):
    def __init__(self, screen):
        super(Enemy, self).__init__()
        self.screen = screen
        self.type = Enemy_Type
        self.direction = random.randint(0, 3)
        self.image = pygame.image.load([f'.images/enemy/{file}' for file in os.listdir('.images/enemy/')][self.direction])
        self.rect = self.image.get_rect()
        self.speed = Enemy_Speed
        self.is_alive = True
        self.is_hit_wall = False
        self.bullets = pygame.sprite.Group()
        self.terminal = float(100)

    def random_turn(self):
        self.is_hit_wall = False
        directions = [0, 1, 2, 3]
        directions.remove(self.direction)
        self.direction = directions[random.randint(0, 1)]
        self.terminal = float(random.randint(40*2, 40*8))
        self.image = pygame.image.load([f'.images/enemy/{file}' for file in os.listdir('.images/enemy/')][self.direction])

    def random_shot(self):
        shot_prob = random.random()
        if shot_prob < Random_Shot_Prob:
            self.__remove_bullets()
            if not self.is_alive or len(self.bullets) >= 3:
                return
            bullet = Bullet(screen=self.screen)
            bullet.direction = self.direction
            if self.direction == LEFT:
                bullet.rect.right = self.rect.left
                bullet.rect.centery = self.rect.centery
            elif self.direction == RIGHT:
                bullet.rect.left = self.rect.right
                bullet.rect.centery = self.rect.centery
            elif self.direction == UP:
                bullet.rect.bottom = self.rect.top
                bullet.rect.centerx = self.rect.centerx
            elif self.direction == DOWN:
                bullet.rect.top = self.rect.bottom
                bullet.rect.centerx = self.rect.centerx
            self.bullets.add(bullet)

    def update(self):
        if self.direction == LEFT:
            self.rect.x -= self.speed
        elif self.direction == RIGHT:
            self.rect.x += self.speed
        elif self.direction == UP:
            self.rect.y -= self.speed
        elif self.direction == DOWN:
            self.rect.y += self.speed

        self.random_shot()
        if self.terminal <= 0:
            self.random_turn()
        else:
            self.terminal -= self.speed

    def __remove_bullets(self):
        for bullet in self.bullets:
            rect = bullet.rect
            if rect.bottom <= 0 or rect.top >= SCREEN_RECT.bottom or rect.right <= 0 or rect.left >= SCREEN_RECT.right:
                self.bullets.remove(bullet)
                bullet.kill()

    def hit_wall(self):
        can_turn = False
        if self.direction == LEFT and self.rect.left <= 0:
            can_turn = True
            self.rect.left = 2
        elif self.direction == RIGHT and self.rect.right >= SCREEN_RECT.right - 1:
            can_turn = True
            self.rect.right = SCREEN_RECT.right - 2
        elif self.direction == DOWN and self.rect.bottom >= SCREEN_RECT.bottom - 1:
            can_turn = True
            self.rect.bottom = SCREEN_RECT.bottom - 2
        elif self.direction == UP and self.rect.top <= 0:
            can_turn = True
            self.rect.top = 2
        if can_turn:
            self.random_turn()

    def out_of_wall(self, wall):
        if self.direction == LEFT:
            self.rect.left = wall.rect.right + 2
        elif self.direction == RIGHT:
            self.rect.right = wall.rect.left - 2
        elif self.direction == UP:
            self.rect.top = wall.rect.bottom + 2
        elif self.direction == DOWN:
            self.rect.bottom = wall.rect.top - 2

    def boom(self):
        for sub_boom in ['.images/boom/' + file for file in os.listdir('.images/boom')]:
            self.image = pygame.image.load(sub_boom)
            time.sleep(Tank_Boom_Sleep)
            self.screen.blit(self.image, self.image.get_rect())
        super().kill()

    def kill(self):
        self.is_alive = False
        t = Thread(target=self.boom)
        t.start()


class Bullet(pygame.sprite.Sprite):
    def __init__(self, screen):
        super().__init__()
        self.image = pygame.image.load('.images/bullet/bullet1.png')
        self.screen = screen
        self.direction = None
        self.speed = Bullet_Speed
        self.rect = self.image.get_rect()

    def update(self):
        if self.direction == LEFT:
            self.rect.x -= self.speed
        elif self.direction == RIGHT:
            self.rect.x += self.speed
        elif self.direction == UP:
            self.rect.y -= self.speed
        elif self.direction == DOWN:
            self.rect.y += self.speed


class Wall(pygame.sprite.Sprite):
    def __init__(self, wall_type, x_y, screen):
        super(Wall, self).__init__()
        self.image = pygame.image.load([f'.images/walls/{file}' for file in os.listdir('.images/walls/')][wall_type])
        self.rect = self.image.get_rect()
        self.type = wall_type
        self.screen = screen
        self.location = x_y
        self.life = Wall_Life
        self.rect.x = x_y[1] * BOX_SIZE
        self.rect.y = x_y[0] * BOX_SIZE

    def boom(self):
        for sub_boom in ['.images/boom/' + file for file in os.listdir('.images/boom')]:
            self.image = pygame.image.load(sub_boom)
            time.sleep(Wall_Boom_Sleep)
            self.screen.blit(self.image, self.rect)
        super().kill()
        return self.location

    def kill(self):
        self.life -= 1
        if self.life<=0:
            t = Thread(target=self.boom)
            t.start()
        return self.location
