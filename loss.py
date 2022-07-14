import numpy as np


class Loss:
	def __init__(self):
		pass

	def f(self, y_pred, y_true):
		pass

	def df(self, y_pred, y_true):
		pass


class MSE(Loss):
	def __init__(self):
		pass

	def f(self, y_pred, y_true):
		return np.mean(np.power(y_pred - y_true, 2))

	def df(self, y_pred, y_true):
		return 2 * (y_pred - y_true) / len(y_true)