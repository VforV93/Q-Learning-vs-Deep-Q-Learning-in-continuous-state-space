# Q-Learning-vs-Deep-Q-Learning-in-continuous-state-space
Autonomous and Adaptive Systems Exam Project

The project's purpose is comparing, from a performance point of view, two of the best known algorithms in the field of reinforcement learning: Q-Learning vs Deep Q-Learning. Can the discretization of the space "almost" guarantee the same performance of a deep q learning approach in some cases? From this project the answer is "Yes" however, as the results show, the context in which the discretization of the continuous state space is applied shows that the performance could be comparable in terms of accuracy and score average but not in terms of learning speed. The Deep Q-learning Agent is always faster than the Q-Learning Agent.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Installation for Windows

```
Python Version 3.6
Anaconda 1.9.12
```

### Installing

Create a new enviroment(ex: conda env)
```
conda create -n <env_name> python=3.6
```

Install all the packeges listed in requirements.txt
```
conda install --yes --file requirements.txt
```
