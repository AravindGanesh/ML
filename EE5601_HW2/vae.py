import numpy as np
import matplotlib.pyplot as plt

def show_digit(x): # x - 196 dim vector
    x = np.reshape(x, (14,14))
    plt.imshow(x, cmap='gray')
    plt.show()

class VAE(object):
    def __init__(self, input_dim, latent_dim, learn_rate):
        self.input_dim = input_dim
        self.learn_rate = learn_rate
        self.latent_dim = latent_dim
        self.output_dim = input_dim
        # intialize weights
        # Weights and Biases of input to mean
        self.U = np.random.normal(0, 1, (self.latent_dim, self.input_dim))
        self.u0 = np.random.normal(0, 1, self.latent_dim)
        # weights and biases of input to covariance
        self.V = np.random.normal(0, 1, (self.latent_dim, self.input_dim))
        self.v0 = np.random.normal(0, 1, self.latent_dim)
        # weights and biases of latent variable to output
        self.W = np.random.normal(0, 1, (self.output_dim, self.latent_dim))
        self.w0 = np.random.normal(0, 1, self.output_dim)
        

    def sigmoid(self, t):
        return 1/(1  + np.exp(-t))

    def dsigmoid(self, t):
        sigt = self.sigmoid(t)
        return sigt*(1-sigt)
   
    def tanh(self, t):
        return (2*self.sigmoid(2*t) - 1)
    
    def dtanh(self, t):
        return (1 - self.tanh(x)**2)

    def encoder(self, x):
        mean = self.tanh(np.dot(x, self.U.T) + self.u0)
        log_covar = self.tanh(np.dot(x, self.V.T) + self.v0)
        return [mean, log_covar] 

    def decoder(self, z):
        return self.sigmoid(np.dot(z, self.W.T) + self.w0)

    def reduce_loss(self, X):
        self.N = len(X)
        #print(X.shape)
        # derivative of loss w.r.t weights and biases
        dLoss_U, dLoss_u0 = np.zeros_like(self.U), np.zeros_like(self.u0)
        dLoss_V, dLoss_v0 = np.zeros_like(self.V), np.zeros_like(self.v0)
        dLoss_W, dLoss_w0 = np.zeros_like(self.W), np.zeros_like(self.w0)
        # mean.shape, covar.shape = (N,k) 
        [mean, log_covar] = self.encoder(X) # mean, log_covar
        covar = np.exp(log_covar)
        e = np.random.normal(0, 1, mean.shape)
        Z = (covar* e) + mean
        # Y.shape = (N, d)
        Y = self.decoder(Z)
        # dY = dsigmoid(np.dot(z, W.T) + w0)
        # dY computed in this way and stored to avoid repeated calling of sigmoid function
        dY = Y * (1-Y)
        # similarly, derivative of tanh = 1 - tanh**2
        dmean = 1 - mean**2
        dlog_covar = 1 - log_covar**2
        self.loss = np.linalg.norm(Y-X)**2 + 0.5*np.sum((covar - mean**2 -1 - log_covar))
        # y_delta.shape (N,d)
        y_delta = 2* (Y-X) * dY
        # mean.shape (N,k)
        mean_delta = np.multiply(np.matmul(y_delta, self.W), dmean) 
        dKL_U = np.sum(mean*dmean, axis=1) # shape = (N,)
        covar_delta = np.matmul(y_delta, self.W) * e * covar * dlog_covar
        dKL_V = 0.5 * np.sum((covar-1)*dlog_covar, axis=1)

        dLoss_W = np.matmul(y_delta.T, Z)
        dLoss_w0 = np.sum(y_delta, axis=0)
        dLoss_U = np.matmul((mean_delta.T + dKL_U) , X)
        dLoss_u0 = np.sum(mean_delta.T + dKL_U, axis=1)
        dLoss_V = np.matmul((covar_delta.T + dKL_V) , X)
        dLoss_v0 = np.sum((covar_delta.T + dKL_V), axis=1)
        # update weights 
        self.U  -= self.learn_rate * dLoss_U
        self.u0 -= self.learn_rate * dLoss_u0
        self.V  -= self.learn_rate * dLoss_V
        self.v0 -= self.learn_rate * dLoss_v0
        self.W  -= self.learn_rate * dLoss_W
        self.w0 -= self.learn_rate * dLoss_w0
    
    
    def train(self, x_train, epochs, shuffle=True): 
        epoch = 1
        while(epoch <= epochs):
            if shuffle:
                indices = np.arange(len(x_train))
                np.random.shuffle(indices)
                x_train = x_train[indices]
            self.reduce_loss(X=x_train)
            print('Epoch: ', epoch, 'Loss: ', self.loss)
            # show_digit(x_train[0])
            z = np.random.normal(0,1, self.latent_dim)
            if epoch%5==0: show_digit(self.decoder(z))
            epoch += 1
        print('Done Training')