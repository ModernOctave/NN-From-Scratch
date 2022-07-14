import numpy as np


class Activation:
	def __init__(self):
		pass

	def f(self, x):
		pass

	def df(self, x):
		pass


class Sigmoid(Activation):
	def __init__(self):
		pass

	def f(self, x):
		return 1 / (1 + np.exp(-x))

	def df(self, x):
		return self.f(x) * (1 - self.f(x))