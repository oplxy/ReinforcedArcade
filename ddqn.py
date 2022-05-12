import pygame
from pygame.locals import *
from gym import spaces

import math
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
from segment_tree import MinSegmentTree, SumSegmentTree

# screen setup
# screen = pygame.display.set_mode((1000, 600))
pygame.display.set_caption('platformer')
# picture import
brickpic = pygame.image.load('brick.bmp')
bluepic = pygame.image.load('Blue.bmp')
keypic = pygame.image.load('whitesquare.bmp')
spikepic = pygame.image.load('spike.bmp')
r = 38
brickpic = pygame.transform.scale(brickpic, (r, r))
bluepic = pygame.transform.scale(bluepic, (r, r))
keypic = pygame.transform.scale(keypic, (r, r))
spikepic = pygame.transform.scale(spikepic, (r, r))

rendering = False
framerate = 30
next_stage = False
window_width, window_height = 1000, 600

spikegroup = pygame.sprite.Group()
brickgroup = pygame.sprite.Group()
#screen = pygame.display.set_mode((1000, 600))

action_list = []
text = ''
seed_record = ""
# classes
class Player(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        # image
        self.image = bluepic

        # initial value & rect
        self.rect = self.image.get_rect()
        self.rect.x = 0
        self.rect.y = 150
        self.xpos = 0
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
        self.finish = False

        # player state
        self.isalive = True
        generate_stage()

    def nextframe(self, c):
        global text
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
        text += c+' '

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

    def update(self, action):
        global next_stage
        self.nextframe(f'{action:05b}')
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
                self.xpos += self.rect.x-brick.rect.x+r
                self.rect.x = brick.rect.x - r

            elif self.left and not self.wall and abs(rely + 0.001) <= r and abs(relx + r) < abs(self.xvel) + 0.01:
                self.wall = True
                self.xpos += self.rect.x-brick.rect.x-r
                self.rect.x = brick.rect.x + r
        for spike in spikegroup:
            relx = spike.rect.x - self.rect.x
            rely = spike.rect.y - self.rect.y
            if abs(rely) < r - 1 and abs(relx) < r - 1:
                self.isalive = False
                break
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
            self.wall=True
            self.xpos += -self.rect.x-20
            self.rect.x = -20
        # xvel process
        if not self.wall:
            self.rect.x = self.rect.x + self.xvel
            self.xpos = self.xpos + self.xvel
        # yvel process
        # print('yvel', self.yvel)
        self.rect.y = self.rect.y + self.yvel
        # blit
        #screen.blit(self.image, (self.rect.x, self.rect.y))
        self.onplatform = False
        self.wall = False
        if self.xpos > 10200 or not self.isalive:
            self.finish = True


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
        #screen.blit(self.image, (self.rect.x, self.rect.y))
        pass


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
    global seed_record
    fill = set()
    filling = True
    brickgroup.empty()
    spikegroup.empty()
    l = 0
    for x in range(random.randint(5, 30)):#TODO 5~30 => 1(not activate)
        a, b = random.randint(0, 26), random.randint(0, 3)
        i = 1
        d = 1
        while str(a).zfill(2) + str(b) in fill:
            a += i * d
            i += 1
            d = -d
            if a < 0 or a > 26:
                filling = False
                break
        if filling:
            fill.add(str(a).zfill(2) + str(b))
        filling = True
    l = len(fill)
        #TODO
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
                if a < 0 or a > 26:
                    filling = False
                    break
            if filling:
                fill.add(str(a).zfill(2) + "0")
                spikegroup.add(Spike(a * 38 + 19, 500, spikepic))  # 526 = 13*39+19
            filling = True
    print(l,fill)
    seed_record+=(str(l)+" "+"".join(fill)+'f')
class CustomEnv(gym.Env):
    def __init__(self, env_config={}):
        # self.observation_space = gym.spaces.Box()
        # self.action_space = gym.spaces.Box()
        pygame.init()
        self.player = Player()
        self.action_space = spaces.Discrete(32)
        self.observation_space = spaces.Box(low=np.zeros((122,)), high=np.zeros((122,)), dtype=np.float64)
        self.deltax = 0
        self.deltay = 0
        self.frame = 0

    def init_render(self):
        self.screen = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()

    def reset(self):
        self.__init__()
        re = np.concatenate((
             np.array([self.player.rect.x, self.player.rect.y]),
             np.concatenate((np.concatenate([np.array([brick.rect.x, brick.rect.y]) for brick in brickgroup]),
                             np.empty((60 - len(brickgroup)*2,)))) if len(brickgroup) else np.empty((60,)),
             np.concatenate((np.concatenate([np.array([spike.rect.x, spike.rect.y]) for spike in spikegroup]),
                             np.empty((60 - len(spikegroup)*2,)))) if len(spikegroup) else np.empty((60,)))
            )
        return re

    def step(self, action):
        formery = self.player.rect.y
        formerx = self.player.xpos
        self.frame += 1
        for brick in brickgroup:
            brick.update()
        for spike in spikegroup:
            spike.update()
        if self.player.isalive:
            self.player.update(action)
        self.deltax = self.player.xpos - formerx
        self.deltay = self.player.rect.y - formery
        returner = np.concatenate((
             np.array([self.player.rect.x, self.player.rect.y]),
             np.concatenate((np.concatenate([np.array([brick.rect.x, brick.rect.y]) for brick in brickgroup]),
                             np.empty((60 - len(brickgroup)*2,)))) if len(brickgroup) else np.empty((60,)),
             np.concatenate((np.concatenate([np.array([spike.rect.x, spike.rect.y]) for spike in spikegroup]),
                             np.empty((60 - len(spikegroup)*2,)))) if len(spikegroup) else np.empty((60,)))
            ), \
                   (-1 if self.player.xpos == formerx and self.player.rect.y == formery else\
                       (2 if self.player.finish and self.player.isalive else 0)
                    - (5 if not self.player.isalive else 0))+(self.player.xpos-5*self.frame)/50,\
                   self.player.finish, {}
        return returner

    def render(self):
        self.rendering = True
        #screen.fill((0, 0, 0))
        if self.player.right:
            #screen.blit(keypic, (76, 38))
            pass
        if self.player.left:
            #screen.blit(keypic, (0, 38))
            pass
        brickgroup.update()
        spikegroup.update()
        if self.player.isalive:
            self.player.update()
        pygame.display.update()

    def generate_stage(self):
        generate_stage()
        # fill = set()
        # brickgroup.empty()
        # spikegroup.empty()
        # for x in range(random.randint(5, 30)):
        #     a, b = random.randint(0, 26), random.randint(0, 3)
        #     i = 1
        #     d = 1
        #     while str(a).zfill(2) + str(b) in fill:
        #         a += i * d
        #         i += 1
        #         d = -d
        #     fill.add(str(a).zfill(2) + str(b))
        # for i in fill:
        #     brickgroup.add(Brick(int(i[:2]) * 38 + 19, 500 - int(i[2:]) * 39, brickpic))
        # if spiking:
        #     for x in range(random.randint(2, 10)):
        #         a = random.randint(2, 24)
        #         i = 1
        #         d = 1
        #         while str(a).zfill(2) + "0" in fill:
        #             a += i * d
        #             i += 1
        #             d = -d
        #         fill.add(str(a).zfill(2) + "0")
        #         spikegroup.add(Spike(a * 38 + 19, 500, spikepic))  # 526 = 13*39+19


###################################################################################################
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)

class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
            self,
            env: gym.Env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        self.trainframe = 0

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        global seed, seed_record
        global text
        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        text += str(seed) + ' '
        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            self.trainframe += 1

            # NoisyNet: removed decrease of epsilon

            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)
            # if episode ends
            if done or self.trainframe == 1000:  # or (self.env.deltax == 0 and self.env.deltay == 0 and action):
                state = self.env.reset()
                scores.append(score)
                score = 0
                action_list.append(text)
                seed = random.randint(1, 999999)
                text = str(seed) + ' '
                seed_record += '\n'
                random.seed(seed)
                self.trainframe = 0
            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)

        self.env.close()

    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state = self.env.reset()
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        # reset
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).gather(  # Double DQN
            1, self.dqn(next_state).argmax(dim=1, keepdim=True)
        ).detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
            epsilons: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()




seed = random.randint(1, 999999)
env = CustomEnv()
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



# parameters
num_frames = 500000
memory_size = 10000
batch_size = 128
target_update = 100
epsilon_decay = 1/2000

# train
agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
agent.train(num_frames, num_frames)#TODO parameter splited into 100 plotting segment

file=open('record.txt','w')
file.write('\n'.join(action_list)) #output side
seed_file=open('seed.txt','w')
seed_file.write(seed_record)