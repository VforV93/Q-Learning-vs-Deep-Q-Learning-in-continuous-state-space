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
    # OR
    # env = CartPole3D(renders=False)

    sBounds = list(zip(env.observation_space.low, env.observation_space.high))

    # cartpole
    bucketS = (5, 5, 15, 5)
    sBounds[1] = [-0.5, 0.5]  # limit cart velocity
    sBounds[3] = [-math.radians(50), math.radians(50)]  # limit pole angular velocity

    # OR

    # cartpole 3D
    # bucketS = (5, 5, 5, 5, 20, 10, 20, 10)
    # sBounds[1] = [-4, 4]  # limit cart velocity on x
    # sBounds[3] = [-4, 4]  # limit cart velocity on y
    # sBounds[5] = [-math.radians(50), math.radians(50)]  # limit pole angular velocity on x
    # sBounds[7] = [-math.radians(50), math.radians(50)]  # limit pole angular velocity on y

    ql = QLearn(env,
                sBounds=sBounds,
                winning_ths=500,
                num_episodes=45000,
                max_step=1000,
                learning_rate=0.5,
                epsilon=0.9,
                bucket_size=bucketS,
                decay_speed=0.99995)

    ql.train(save_path="./results/QLearning/2D/QLearn_CartPole({}).npy".format(ql.num_episodes))


if __name__ == "__main__":
    main()
