#code from https://github.com/ycchen1989/Var-QuantumCircuits-DeepRL/blob/master/Code/ShortestPathFrozenLake.py

import gymnasium as gym
from gymnasium.envs import toy_text

class ShortestPathFrozenLake(toy_text.frozen_lake.FrozenLakeEnv):
	def __init__(self, **kwargs):
		super(ShortestPathFrozenLake, self).__init__(**kwargs)

		for state in range(16): # for all states	self.nS
			for action in range(4): # for all actions self.nA
				my_transitions = []
				for (prob, next_state, _, is_terminal) in self.P[state][action]:
					row = next_state // self.ncol
					col = next_state - row * self.ncol
					tile_type = self.desc[row, col]
					if tile_type == b'H':
						reward = -0.2
					elif tile_type == b'G':
						reward = 1.
					else:
						reward = -0.01

					my_transitions.append((prob, next_state, reward, is_terminal))
				self.P[state][action] = my_transitions
				


