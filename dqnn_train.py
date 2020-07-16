import os, inspect
import gym

from agent.QNet_Final import DQN
from envs.cartPole3D import CartPole3D

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

os.sys.path.insert(0, currentdir + '/env')
os.sys.path.insert(0, currentdir + '/agent')


def main():
    # env = gym.make("CartPole-v1")
    # env._max_episode_steps = 700
    # OR
    env = CartPole3D(renders=False)

    # For the CartPole 2D
    # dql = DQN(env, num_hidden=64, learning_rate=0.00025, epsilon=0.9, epsilon_decay=0.999, winning_ths=300)

    # For the CartPole 3D
    dql = DQN(env, num_hidden=64, learning_rate=0.00025, epsilon=0.2, epsilon_decay=0.9999, winning_ths=500)  # change also the model inside the agent

    dql.train(num_episodes=45000, max_steps=1000)


if __name__ == "__main__":
    main()
