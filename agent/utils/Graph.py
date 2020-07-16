# Credits: https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f
# this class is been made starting from the Plot_res function
# adaptations made
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, goal_threshold, threshold_margin=5, freq=50, title='', update_steps=10, figsize=(18, 5)):
        self.figure, self.ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        self.figure.suptitle(title)
        self.goal_threshold = goal_threshold
        self.threshold_margin = threshold_margin
        self.freq = freq
        self.update_steps = update_steps
        self.values = []

        self.x_maxmin = []
        self.y_max = []
        self.y_mean = []
        plt.show()

    def plot_res(self, vls):
        self.values.extend(vls)
        """Plotting over time."""
        self.ax[0].clear()
        self.ax[0].plot(self.values, label='score per run')

        if len(vls) > 1:
            x = self.x_maxmin[-1] if len(self.x_maxmin)>0 else 0
            self.x_maxmin.append(len(vls)+x)
            self.y_max.append(max(self.values[-self.freq:]))
            self.y_mean.append(np.mean(self.values[-self.freq:]))

            self.ax[0].plot(self.x_maxmin, self.y_max, 'g', label='max reward in {} ep'.format(self.freq))
            self.ax[0].plot(self.x_maxmin, self.y_mean, 'k', label='mean reward in {} ep'.format(self.freq))

        self.ax[0].axhline(self.goal_threshold-self.threshold_margin, c='red', ls='--', label='goal')
        self.ax[0].set_xlabel('Episodes')
        self.ax[0].set_ylabel('Reward')
        x = range(len(self.values))
        self.ax[0].legend()
        # Calculate the trend
        try:
            z = np.polyfit(x, self.values, 1)
            p = np.poly1d(z)
            self.ax[0].plot(x, p(x), "--", label='trend')
        except:
            print('')

        # Plot the histogram of results
        self.ax[1].clear()
        self.ax[1].hist(self.values[-self.freq:])
        self.ax[1].axvline(self.goal_threshold-self.threshold_margin, c='red', label='goal')
        self.ax[1].set_xlabel('Scores per Last {} Episodes'.format(self.freq))
        self.ax[1].set_ylabel('Frequency')
        self.ax[1].legend()
        plt.pause(0.001)

    @staticmethod
    def savefig(file_name, dpi):
        plt.savefig(file_name, dpi=dpi)