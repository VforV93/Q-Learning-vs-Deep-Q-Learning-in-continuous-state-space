from collections import deque
import random
import numpy as np
import os

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from .utils.Graph import Plotter


# Deep Q-learning Agent
class DQN:
    def __init__(self, env, winning_ths=200, num_hidden=24, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.name = 'DQN'
        MEMORY_CAPACITY = 10000

        self.memory = Memory(MEMORY_CAPACITY)
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.learning_rate = learning_rate
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = self._create_model(num_hidden)
        self.winning_ths = winning_ths
        self.graph = Plotter(self.winning_ths, title="{} strategy".format(self.name), update_steps=25)

    def _create_model(self, nh):
        model = Sequential()
        model.add(Dense(nh, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(nh, activation='relu'))
        model.add(Dense(nh, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        model.summary()
        return model

    def memorize(self, s, action, r, next_s, done):
        self.memory.add((s, action, r, next_s, done))

    def replay(self, batch_size):
        batch = self.memory.sample_n(batch_size)
        for s, a, r, next_s, done in batch:
            t = r
            if not done:
                t = r + self.gamma * np.amax(self.model.predict(next_s)[0])
            target_pred = self.model.predict(s)
            target_pred[0][a] = t
            self.model.fit(s, target_pred, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay * self.epsilon
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def act(self, s):
        # Exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        # Exploitation
        actions = self.model.predict(s)
        return np.argmax(actions[0])
    
    def train(self, batch_size=32, max_steps=500, num_episodes=1000, file_name="gym-cartpole-v1.h5"):
        final_r = []
        best_mean = 0
        for iteration in range(num_episodes):
            # reset state at the beginning of each game
            s = self.env.reset()
            s = np.reshape(s, [1, self.state_dim])

            R = 0

            for step in range(max_steps):
                # self.env.render()

                # Action selection
                action = self.act(s)

                # Action execution
                next_s, r, done, _ = self.env.step(action)
                next_s = np.reshape(next_s, [1, self.state_dim])
                R += r

                # Memorize!
                self.memorize(s, action, r, next_s, done)

                s = next_s

                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}[epsilon:{}]".format(iteration, num_episodes, R, self.epsilon))
                    final_r.append(R)
                    break

            # apply replay, train network
            self.replay(batch_size)

            if (iteration % self.graph.update_steps == 0):
                self.graph.plot_res(final_r)
                mean = np.mean(final_r[-50:])
                final_r = []
                if mean > best_mean:
                    if os.path.exists(file_name+"_mean{}.h5".format(best_mean)):
                        os.remove(file_name+"_mean{}.h5".format(best_mean))
                    best_mean = mean
                    self.model.save(file_name+"_mean{}.h5".format(mean))

            if (iteration % 100 == 0):
                self.model.save(file_name)
                self.graph.savefig('{}_graph.png'.format(self.name), dpi=300)

        self.model.save(file_name)
        self.graph.savefig('{}_graph.png'.format(self.name), dpi=300)

    def test(self, file_name, max_steps=500, num_episodes=1000):
        self.epsilon = self.epsilon_min
        self.model.load_weights(file_name)

        win = 0
        avg = 0
        final_r = []

        for iteration in range(num_episodes):
            # reset game
            s = self.env.reset()
            s = np.reshape(s, [1, self.state_dim])

            R = 0
            done = False

            while not done:
                self.env.render()

                # Action selection
                action = self.act(s)

                # Action execution
                next_s, reward, done, _ = self.env.step(action)
                next_s = np.reshape(next_s, [1, self.state_dim])
                R += reward

                s = next_s

                if done or R >= max_steps-1:
                    print("episode: {}/{}, score: {}".format(iteration, num_episodes, R))
                    avg += R
                    final_r.append(R)
                    if R >= self.winning_ths:
                        win += 1.0
                    break

            self.graph.plot_res(final_r)
            final_r = []

        avgR = float(avg) / num_episodes
        print("accuracy: {}".format(win / num_episodes))
        print("avg reward: {}".format(avgR))


class Memory:
    def __init__(self, capacity):
        self.memory = deque([], capacity)
        self.capacity = capacity

    def add(self, sample):
        self.memory.append(sample)

    def sample_n(self, n):
        n = min(n, len(self.memory))
        return random.sample(self.memory, n)
