#  created at 2021 17

# import statement

import time
import pygame
from pygame.locals import *
import random

pygame.init()

# screen setup
screen = pygame.display.set_mode((1000, 600))
pygame.display.set_caption('platformer')
# picture import
brickpic = pygame.image.load('brick.bmp')
bluepic = pygame.image.load('blue.bmp')
keypic = pygame.image.load('whitesquare.bmp')
spikepic = pygame.image.load('spike.bmp')

r = 38
spikepic = pygame.transform.scale(spikepic, (r, r))
brickpic = pygame.transform.scale(brickpic, (r, r))
bluepic = pygame.transform.scale(bluepic, (r, r))
keypic = pygame.transform.scale(keypic, (r, r))

framerate = 30
next_stage = False


# classes
class Player(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        # image
        self.image = bluepic

        # initial value & rect
        self.rect = self.image.get_rect()
        self.rect.x = 50
        self.rect.y = 150
        self.xvel = 0
        self.yvel = 0
        self.g = r * 67.82 / framerate ** 2
        # self.jumph = r * 17.36 / framerate

        # movement state
        self.right = False
        self.left = False
        self.run = False
        self.jump = False
        self.onplatform = False
        self.jumphold = False
        self.jumptimer = 0
        self.runtimer = 0
        self.stop = True
        self.jumpable = True
        self.wall = False

        # player state
        self.isalive = True

        generate_stage()

    def nextframe(self, c):
        """
        (old solution)
        0: No movement
        1: jump
        2: squat / pipe
        3: left
        4: right
        5: right + jump
        6: sprint (fireball)
        7: 5 + 6
        """
        """
        (new solution)
        string or int
        5 digit bit -> 00000
        first  : jump
        second : pipe
        third  : left
        fourth : right
        fifth  : sprint (fireball)
        """
        if isinstance(c, int):
            c = str(c)
        if c[0] == "1":
            if not self.jump and self.jumpable:
                self.jump = True
                self.jumphold = True
                self.jumptimer = 0
        else:
            self.jumphold = False
        if c[1] == "1":
            pass
            # TODO not yet (or never will)
        if c[2] == "1":
            self.left = True
        else:
            self.left = False
        if c[3] == "1":
            self.right = True
        else:
            self.right = False
        if c[4] == "1":
            self.run = True

    def pressbutton(self, event):
        # TODO replace with functions (port for AI)

        if event.key == K_e and (self.right or self.left):
            self.run = True

        if event.key == K_RIGHT or event.key == K_d:
            self.right = True

        if event.key == K_LEFT or event.key == K_a:
            self.left = True

        if event.key == K_w or event.key == K_UP:
            if not self.jump and self.jumpable:
                self.jump = True
                self.jumphold = True
                self.jumptimer = 0

    def unpressbutton(self, event):
        if event.key == K_RIGHT or event.key == K_d:
            self.right = False

        if event.key == K_LEFT or event.key == K_a:
            self.left = False

        if event.key == K_w or event.key == K_UP:
            self.jumphold = False

    def update(self):
        global next_stage
        # movement
        # print('onplat:', self.onplatform, 'jable:', self.jumpable, 'Jtimer:', self.jumptimer)
        if not self.right and not self.left:
            self.runtimer += 1
        else:
            self.runtimer = 0
        if self.runtimer >= 5:
            self.run = False

        self.jumptimer += 1

        # gravity check
        if self.jumptimer == 5 and self.jump:
            self.rect.y -= 1
            if self.jumphold:
                self.yvel = r * -15.21 / framerate
                self.g = r * 34.79 / framerate ** 2
            else:
                self.yvel = r * -17.36 / framerate
                self.g = r * 67.82 / framerate ** 2
            self.jump = False

        # directional movement, running
        if self.right:
            if self.run:
                self.xvel = 9.1 * r / framerate
            else:
                self.xvel = 3.7 * r / framerate
        elif self.left:
            if self.run:
                self.xvel = -9.1 * r / framerate
            else:
                self.xvel = -3.7 * r / framerate
        else:
            self.xvel = 0

        # ground detecting
        for brick in brickgroup:
            relx = brick.rect.x - self.rect.x
            rely = brick.rect.y - self.rect.y
            if not self.onplatform and self.yvel >= 0 and abs(rely - r) <= self.yvel + .001 and abs(relx) < r - 1:
                self.rect.y = brick.rect.y - r + 0.001
                self.onplatform = True
                self.yvel = 0
                self.jumpable = True
            elif not self.onplatform and self.yvel <= 0 and abs(rely + r) <= -self.yvel + .001 and abs(
                    relx) < r - 1 - 1:
                self.rect.y = brick.rect.y + r
                self.yvel = 0
                brick.kill()

            elif self.right and not self.wall and abs(rely + 0.001) <= r and abs(relx - r) < abs(self.xvel) + 0.01:
                self.wall = True
                self.rect.x = brick.rect.x - r
            elif self.left and not self.wall and abs(rely + 0.001) <= r and abs(relx + r) < abs(self.xvel) + 0.01:
                self.wall = True
                self.rect.x = brick.rect.x + r
        for spike in spikegroup:
            relx = spike.rect.x - self.rect.x
            rely = spike.rect.y - self.rect.y
            if abs(rely) < r - 1 and abs(relx) < r - 1:
                self.isalive = False
        # for brick in brickgroup:
        #    if self.onplatform== False and self.ground == False and self.yvel > 0 and abs(brick.rect.y -self.rect.y-r)
        #                                                          <=self.yvel and abs(brick.rect.x - self.rect.x)<= r :
        #        self.rect.y = brick.rect.y-r
        #        self.onplatform = True
        #        self.ground = True
        #        self.yvel = 0
        #        print('a')
        #        break
        # print(self.onplatform)
        if self.rect.y > 500:
            self.rect.y = 501
            self.jumpable = True
            self.onplatform = True
            self.yvel = 0
        elif self.rect.y < 500 and not self.onplatform:
            self.jumpable = False
        if not self.onplatform:
            self.yvel = self.yvel + self.g
            # if over adding
            # if self.yvel >30:
            #    self.yvel = 30
        else:
            self.g = r * 55.88 / framerate ** 2
        if self.rect.x > 1020:
            self.rect.x = -20
            generate_stage()
        elif self.rect.x < -20:
            self.rect.x = 1020
        # xvel process
        if not self.wall:
            self.rect.x = self.rect.x + self.xvel
        # yvel process
        # print('yvel', self.yvel)
        self.rect.y = self.rect.y + self.yvel
        # blit
        screen.blit(self.image, (self.rect.x, self.rect.y))
        self.onplatform = False
        self.wall = False


class Object(pygame.sprite.Sprite):
    def __init__(self, x, y, image):
        pygame.sprite.Sprite.__init__(self)
        # image
        self.image = image

        # initial value & rect
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def update(self):
        screen.blit(self.image, (self.rect.x, self.rect.y))


class Brick(Object):
    def __init__(self, x, y, image):
        super().__init__(x, y, image)

    def update(self):
        super(Brick, self).update()


class Spike(Object):
    def __init__(self, x, y, image):
        super().__init__(x, y, image)

    def update(self):
        super(Spike, self).update()


spiking = True


def generate_stage():
    fill = set()
    brickgroup.empty()
    spikegroup.empty()
    for x in range(random.randint(5, 30)):
        a, b = random.randint(0, 26), random.randint(0, 3)
        i = 1
        d = 1
        while str(a).zfill(2) + str(b) in fill:
            a += i * d
            i += 1
            d = -d
        fill.add(str(a).zfill(2) + str(b))
    for i in fill:
        brickgroup.add(Brick(int(i[:2]) * 38 + 19, 500 - int(i[2:]) * 39, brickpic))
    if spiking:
        for x in range(random.randint(2, 10)):
            a = random.randint(2, 24)
            i = 1
            d = 1
            while str(a).zfill(2) + "0" in fill:
                a += i * d
                i += 1
                d = -d
            fill.add(str(a).zfill(2) + "0")
            spikegroup.add(Spike(a * 38 + 19, 500, spikepic))  # 526 = 13*39+19


spikegroup = pygame.sprite.Group()
brickgroup = pygame.sprite.Group()
brick = Brick(400, 400, brickpic)
spike = Spike(400, 400, spikepic)
brickgroup.add(brick)
player = Player()
run = True
Clock = pygame.time.Clock()
while run:
    Clock.tick(framerate)
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            player.pressbutton(event)
        if event.type == KEYUP:
            player.unpressbutton(event)
        if event.type == QUIT:
            running = False
            pygame.quit()
            exit()
    screen.fill((0, 0, 0))
    if player.right:
        screen.blit(keypic, (76, 38))
    if player.left:
        screen.blit(keypic, (0, 38))
    print(player.rect.x, player.rect.y)
    brickgroup.update()
    spikegroup.update()
    if player.isalive:
        player.update()
    pygame.display.update()
