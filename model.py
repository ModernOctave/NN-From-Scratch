import numpy as np
from layer import Layer
from loss import Loss

class SequentialModel:
	def __init__(self, layers: list[Layer], loss: Loss):
		self.layers = layers
		self.loss = loss

	def predict(self, x: float):
		for layer in self.layers:
			x = layer.forwardpass(x)
		return x

	def fit(self, X, Y, epochs, learning_rate):
		num_samples = len(X)

		for epoch in range(epochs):
			err = 0
			dE_dY = 0
			for x, y in zip(X, Y):
				x, y = np.array([x]), np.array([y])
				for layer in self.layers:
					x = layer.forwardpass(x)

				err += self.loss.f(x, y) / num_samples
				dE_dY = self.loss.df(x, y) / num_samples

				for layer in reversed(self.layers):
					dE_dY = layer.backprop(dE_dY, learning_rate)

			print("Epoch:", epoch, "Error:", err)