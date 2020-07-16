import numpy as np
import random

from .utils.Graph import Plotter
import os


class QLearn:
    def __init__(self,
                 env,
                 sBounds,
                 winning_ths,
                 bucket_size,
                 num_episodes=10000,
                 max_step=700,
                 epsilon=0.4,
                 learning_rate=0.5,
                 decay_speed=0.999
                 ):
        self.name = "QLearn"
        self.env = env
        self.bucket_size = bucket_size
        self.sBounds = sBounds

        self.num_actions = env.action_space.n

        # Q-Table for each state-action pair
        self.q_table = np.zeros(bucket_size + (self.num_actions,))

        # parameters
        self.min_eps = 0.01
        self.epsilon = epsilon
        self.min_learn = 0.1
        self.learning_rate = learning_rate
        self.decay_speed = decay_speed

        self.num_episodes = num_episodes
        self.max_t = max_step

        # Plotting info
        self.winning_ths = winning_ths
        self.graph = Plotter(self.winning_ths, title="{} strategy".format(self.name), update_steps=50)

    def train(self, save_path, load=None):
        dir = save_path.split('/')
        try:
            os.makedirs(save_path[:-len(dir[-1:][0])])
        except OSError:
            pass

        if load is not None:
            try:
                self.q_table = np.load(load)
            except OSError:
                print("file not found")
                raise

        np.save(save_path, self.q_table)

        learning_rate = self.learning_rate
        explore_rate = self.epsilon
        gamma = 0.99  # discount value

        best_mean = 0
        final_r = []

        for episode in range(self.num_episodes):

            # Reset the environment
            obs = self.env.reset()

            # the initial state
            s = self._from_s_to_bucket(obs)

            for t in range(self.max_t):
                # self.env.render()

                # Action selection
                action = self._choose_action(s, explore_rate)

                # Action execution
                obs, reward, done, _ = self.env.step(action)

                # Observation
                next_s = self._from_s_to_bucket(obs)

                # Update Q
                best_future_q = np.amax(self.q_table[next_s])
                self.q_table[s + (action,)] += learning_rate * (
                        reward + gamma * best_future_q - self.q_table[s + (action,)])

                s = next_s

                if done or t >= self.max_t-1:
                    final_r.append(t)
                    print("Episode {} is terminated in {} steps[Explorate rate: {}, lr: {}]".format(episode, t, explore_rate, learning_rate))
                    break

            if(episode % self.graph.update_steps == 0):
                self.graph.plot_res(final_r)
                mean = np.mean(final_r[-50:])  # mean based on the last 50 episodes score
                final_r = []
                if mean > best_mean:
                    if os.path.exists("{}_mean{}.npy".format(self.name, best_mean)):
                        os.remove("{}_mean{}.npy".format(self.name, best_mean))
                    best_mean = mean
                    np.save("{}_mean{}.npy".format(self.name, best_mean), self.q_table)

            if(episode % 100 == 0):
                np.save(save_path, self.q_table)
                self.graph.savefig('{}_graph.png'.format(self.name), dpi=300)

            # Update parameters
            explore_rate = max(self.min_eps, explore_rate * self.decay_speed)
            learning_rate = max(self.min_learn, learning_rate * self.decay_speed)

        # Final saving of the Q-table
        np.save(save_path, self.q_table)
        self.graph.savefig('{}_graph.png'.format(self.name), dpi=300)

    def test(self, path, num_episodes, max_steps=700):
        try:
            self.q_table = np.load(path)
        except OSError:
            print("file not found")
            raise

        win = 0
        avg = 0
        final_r = []

        for episode in range(num_episodes):

            # Reset the environment
            obs = self.env.reset()

            # the initial state
            s = self._from_s_to_bucket(obs)

            for t in range(max_steps):
                # self.env.render()

                # Action selection
                action = self._choose_action(s, explore_rate=0)

                # Action execution
                obs, reward, done, _ = self.env.step(action)

                # Observation
                next_s = self._from_s_to_bucket(obs)
                s = next_s

                if done or t >= max_steps-1:
                    print("Episode {} is terminated in {} steps".format(episode, t))
                    avg += t+1
                    final_r.append(t)
                    if t >= self.winning_ths:
                        win += 1.0
                    break

            self.graph.plot_res(final_r)
            final_r = []

        avgR = float(avg) / num_episodes
        print("accuracy: {}".format(win / num_episodes))
        print("avg reward: {}".format(avgR))

    def _choose_action(self, state, explore_rate):
        # Exploration
        if random.random() < explore_rate:
            action = random.randint(0, self.num_actions-1)
        else:  # Exploitation
            action = np.argmax(self.q_table[state])
        return action

    def _from_s_to_bucket(self, state):
        bucket_index_components = []

        for i in range(len(state)):
            left_bound = self.sBounds[i][0]  # Es -4.8
            right_bound = self.sBounds[i][1]  # Es 4.8
            interval_dimension = self.sBounds[i][1] - self.sBounds[i][0]  # and 9.6
            if state[i] <= left_bound:
                bucket_index = 0
            elif state[i] >= right_bound:
                bucket_index = self.bucket_size[i] - 1
            else:
                # if state[i] is inside the range [left_bound, right_bound]
                bucket_index = 0
                bucket_dim = interval_dimension / self.bucket_size[i]
                left_limit = left_bound
                while state[i] > (left_limit + bucket_dim):
                    bucket_index += 1
                    left_limit += bucket_dim
            bucket_index_components.append(bucket_index)
        return tuple(bucket_index_components)
