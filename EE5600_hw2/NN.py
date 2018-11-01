import numpy as np

class NN(object):
	def __init__(self, hidden_dim=2, learn_rate=0.01):
		self.learn_rate = learn_rate
		self.input_dim = 2
		self.hidden_dim = hidden_dim
		self.output_dim = 1
 
		self.A = np.random.normal(0, 1, (self.hidden_dim, self.input_dim))
		self.B = np.random.normal(0, 1, self.hidden_dim)
		self.a0 = np.random.normal(0, 1, self.hidden_dim)
		self.b0 = np.random.normal(0, 1, 1)
		
	def sigmoid(self, t):
		return 1/(1  + np.exp(-t))
	
	def dsigmoid(self, t):
		return self.sigmoid(t)*(1 - self.sigmoid(t))
	
	def hidden_layer(self, x):
		self.z = self.sigmoid(np.dot(self.A, x) + self.a0) 
		return self.z
	
	def forward_pass(self, x):
		self.y_hat = self.sigmoid(np.dot(self.B, self.hidden_layer(x)) + self.b0)
		return self.y_hat
	
	def back_propogate(self, X, Y, Y_hat):
		SSE_a, SSE_b   = np.zeros_like(self.A), np.zeros_like(self.B)
		z_bias, y_bias = np.zeros_like(self.a0), np.zeros_like(self.b0)
		for i, x in enumerate(X):
			z = self.hidden_layer(x)
			y_error = Y[i] - Y_hat[i]
			y_delta = -2* y_error * self.dsigmoid(np.dot(self.B, z) + self.b0)
			s = self.dsigmoid(np.dot(self.A,x) + self.a0) * self.B * y_delta
			SSE_b += y_delta*z
			SSE_a += np.tensordot(x,s, axes=0)
			y_bias += y_delta
			z_bias += s
		# update the weights and biases
		self.A -= self.learn_rate * SSE_a
		self.B -= self.learn_rate * SSE_b
		self.a0 -= self.learn_rate * s
		self.b0 -= self.learn_rate * y_delta
		
	def train(self, x_train, y_train, epochs, shuffle=True): 
		if shuffle:
			indices = np.arange(N)
			np.random.shuffle(indices)
			x_train, y_train = x_train[indices], y_train[indices]

		epoch = 1
		while(epoch <= epochs):
			y_hat = np.array([self.forward_pass(x) for x in x_train])
			self.back_propogate(x_train, y_train, y_hat)
			print('Epoch: ', epoch, 'Loss: ', self.loss(y_train, y_hat))
			epoch += 1

	def loss(self, y_train, y_hat):
		return np.mean((y_train - y_hat)**2)
	
	def predict(self, test_x):
#         if test_x.shape == (2,): test_x = np.reshape(test_x, (1,2))
		y_hats = np.array([self.forward_pass(x) for x in test_x])
		y_outs = int(y_hats > 0.5)
		return y_outs
	
#     def load_model(self):
#         # something here
#     def get_weights(self):
#         # some code here
#     def load_weights(self):
#         # some code here
#     def save_model(self):
#         # some code here