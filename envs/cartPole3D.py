import os, inspect
# Credits:
# Environment freely available from pybullet SDK
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/cartpole_bullet.py
# very tiny changes done
# See also:
# - https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
# - https://github.com/matpalm/cartpoleplusplus/blob/master/bullet_cartpole.py

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import logging

logger = logging.getLogger(__name__)


class CartPole3D(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, renders=True):
        # start the bullet physics server
        self._renders = renders
        if (renders):
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.x_threshold = 4.8
        self.y_threshold = 4.8
        self.theta_threshold_radians = 15 * 2 * math.pi / 360

        self.observationDim = 8
        p.resetDebugVisualizerCamera(1, 0, -41, [0, -1.5, 0.7])
        observation_high = np.array([np.finfo(np.float32).max] * self.observationDim)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.observation_space.low[0] = -self.x_threshold
        self.observation_space.high[0] = self.x_threshold

        self.observation_space.low[2] = -self.y_threshold
        self.observation_space.high[2] = self.y_threshold

        self.observation_space.low[4] = -0.418
        self.observation_space.high[4] = 0.418

        self.observation_space.low[4] = -0.418
        self.observation_space.high[4] = 0.418

        self._seed()
        self.reset()
        self.viewer = None
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        p.stepSimulation()

        self.state = p.getJointState(self.cartpole, 0)[0:2] + p.getJointState(self.cartpole, 1)[0:2]

        dv = 0.1
        deltav = [-.3 * dv, .3 * dv, -.3 * dv, .3 * dv][action]

        if (action == 0 or action == 1):
            p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, targetVelocity=(deltav + self.state[1]))
        else:
            p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, targetVelocity=(deltav + self.state[3]))

        self.state = p.getJointState(self.cartpole, 0)[0:2] + p.getJointState(self.cartpole, 1)[0:2] + \
                    p.getJointState(self.cartpole, 2)[0:2] + p.getJointState(self.cartpole, 3)[0:2]

        x, x_dot, y, y_dot, theta, theta_dot, theta1, theta1_dot = self.state
        isDone = x < -self.x_threshold \
               or x > self.x_threshold \
               or y < -self.y_threshold \
               or y > self.y_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians \
               or theta1 < -self.theta_threshold_radians \
               or theta1 > self.theta_threshold_radians

        done = isDone
        reward = 1.0
        return np.array(self.state), reward, done, {}

    def reset(self):

        p.resetSimulation()
        self.cartpole = p.loadURDF("cart-pole-Hard.urdf", [0, 0, 0])

        self.wallR = p.loadURDF("wall.urdf", [2.9, 0, 0])
        self.wallT = p.loadURDF("wall.urdf", [0, 2.9, 0], baseOrientation=[0, 0, 1, 1])
        self.wallB = p.loadURDF("wall.urdf", [0, -2.9, 0], baseOrientation=[0, 0, 1, 1])
        self.wallL = p.loadURDF("wall.urdf", [-2.9, 0, 0])
        p.loadURDF("plane.urdf", [0, 0, -0.05])
        self.timeStep = 0.01

        p.setGravity(0, 0, -10)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # I can also decide to set a random starting angles to my pole
        angle1 = self.np_random.uniform(low=-0.01, high=0.01, size=(1,))
        angle2 = self.np_random.uniform(low=-0.01, high=0.01, size=(1,))

        p.resetJointState(self.cartpole, 2, 0)
        p.resetJointState(self.cartpole, 3, 0)

        self.state = p.getJointState(self.cartpole, 0)[0:2] + p.getJointState(self.cartpole, 1)[0:2] + \
                     p.getJointState(self.cartpole, 2)[0:2] + p.getJointState(self.cartpole, 3)[0:2]

        fromp = [2.4, -0.2, 0]
        top = [2.4, 0.2, 0]
        p.addUserDebugLine(fromp, top)

        fromp = [-2.4, -0.2, 0]
        top = [-2.4, 0.2, 0]
        p.addUserDebugLine(fromp, top)

        fromp = [-0.2, 2.4, 0]
        top = [0.2, 2.4, 0]
        p.addUserDebugLine(fromp, top)

        fromp = [-0.2, -2.4, 0]
        top = [0.2, -2.4, 0]
        p.addUserDebugLine(fromp, top)

        return np.array(self.state)

    def render(self, mode='human', close=False):
        return
