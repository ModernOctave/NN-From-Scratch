import numpy as np
from activation import Activation


class Layer:
	def __init__(self, num_inputs: int, num_outputs: int):	
		self.input = None
		self.output = None    

	def forwardpass(self, input: np.ndarray):
		raise NotImplementedError

	def backprop(self, dE_dY: np.ndarray, learning_rate: float):
		raise NotImplementedError

class DenseLayer(Layer):
	def __init__(self, num_inputs, num_outputs):
		self.weights = np.random.rand(num_inputs + 1, num_outputs)
		self.input = None
		self.output = None

	def forwardpass(self, input: np.ndarray):
		self.input = np.hstack([input, np.ones((1,1))])
		self.output = np.matmul(self.input, self.weights)
		return self.output                                      

	def backprop(self, dE_dY: np.ndarray, learning_rate: float):
		dE_dW = np.matmul(self.input.T, dE_dY)
		dE_dX = np.matmul(dE_dY, self.weights[:-1][:].T)
		self.weights -= dE_dW * learning_rate
		return dE_dX

class ActivationLayer(Layer):
	def __init__(self, activation: Activation):
		self.input = None
		self.output = None
		self.activation = activation

	def forwardpass(self, input: np.ndarray) -> np.ndarray:
		self.input = input
		self.output = self.activation.f(self.input)
		return self.output

	def backprop(self, dE_dY: np.ndarray, learning_rate: float):
		return self.activation.df(self.input) * dE_dY