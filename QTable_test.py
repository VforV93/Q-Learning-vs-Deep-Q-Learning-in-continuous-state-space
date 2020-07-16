import inspect
import math
import os

import gym
from agent.Qlearn import QLearn
from envs.cartPole3D import CartPole3D

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir + '/env')
os.sys.path.insert(0, currentdir + '/agent')


def main():
    env = gym.make("CartPole-v1")
    env._max_episode_steps = 700
    # env = CartPole3D(renders=False)

    sBounds = list(zip(env.observation_space.low, env.observation_space.high))

    # cartpole
    bucketS = (5, 5, 15, 5)
    sBounds[1] = [-0.5, 0.5]
    sBounds[3] = [-math.radians(50), math.radians(50)]

    # OR

    # cartpole 3D
    # bucketS = (5, 5, 5, 5, 20, 10, 20, 10)
    # sBounds[1] = [-4, 4]  # limit cart velocity on x
    # sBounds[3] = [-4, 4]   # limit cart velocity on y
    # sBounds[5] = [-math.radians(50), math.radians(50)]  # limit pole angular velocity on x
    # sBounds[7] = [-math.radians(50), math.radians(50)]  # limit pole angular velocity on y

    ql = QLearn(env,
                sBounds=sBounds,
                winning_ths=300,
                num_episodes=100,
                max_step=700,
                bucket_size=bucketS,
                decay_speed=0.999)

    ql.test(path="./results/QLearning/2D/run 35000/BEST_TABLE_QLearn_mean341.52.npy",
            num_episodes=100,
            max_steps=700)


if __name__ == "__main__":
    main()
