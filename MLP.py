import numpy as np

class MLP:
    def __init__(self, layers, activation='sigmoid', lr=0.1, bias=False):      
        # set layers dims
        self.layers = layers
        
        # set learning rate
        self.lr = lr
        
        self.bias = bias
        
        # set activation function
        self.activation_function = self.sigmoid if activation == 'sigmoid' else self.tanh
        self.activation_derivation = self.sigmoid_derivative if activation == 'sigmoid' else self.tanh_derivative
        
        # init wieghts
        self.W = self.init_weights(layers)
    
    def init_weights(self, layers):
        weights = []
        # init hidden layers weights
        for i in np.arange(0, len(layers) - 2):
            # create weights  matricies with size = #ith layer nodes + 1 (bias) x #(i+1)th layer nodes + 1 (bias)
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            
            # normalize the weights
            weights.append(w / np.sqrt(layers[i]))
        
        # init output layer weights
        w = np.random.randn(layers[-2] + 1, layers[-1])
        weights.append(w / np.sqrt(layers[-2]))
        
        return weights
    
    def fit(self, X, y, epochs=1000, eval_step=100):

        X = np.c_[X, np.ones((X.shape[0]))]
        
        for epoch in np.arange(0, epochs):

            for (x, target) in zip(X, y):
                self.propagate(x, target)

            # display a training update
            if epoch == 0 or (epoch + 1) % eval_step == 0:
                loss = self.MSE(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))
                
    def propagate(self, x, y):
        # FEEDFORWARD
        A = self.feed_forward_pass(x, self.W)
        
        # BACKPROPAGATION
        D = self.back_propagation_pass(A, y, self.W)
        
        # UPDATE WEIGHTS
        self.W = self.update_weights(A, self.W, D, self.lr)
        
    
    def feed_forward_pass(self, x, w):
        A = [np.atleast_2d(x)]
        
        for layer in np.arange(0, len(w)):
            net = A[layer].dot(self.W[layer])
            out = self.activation_function(net)

            A.append(out)
        return A
    
    def back_propagation_pass(self, A, y, w):
        error = A[-1] - y

        D = [error * self.activation_derivation(A[-1])]

        for layer in np.arange(len(A) - 2, 0, -1):

            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.activation_derivation(A[layer])
            D.append(delta)
            
        return D
    
    def update_weights(self, A, W, D, lr):
        D = D[::-1]
        
        for layer in np.arange(0, len(W)):
            W[layer] += -lr * A[layer].T.dot(D[layer])
        
        return W
    
    # ACTIVATION FUNCTIONS 
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1.-np.tanh(x)**2 
    
    
    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
            
        # Feed forward pass
        for layer in np.arange(0, len(self.W)):
            p = self.activation_function(np.dot(p, self.W[layer]))

        return p
    
    # Loss function
    def MSE(self, X, y):
        targets = np.atleast_2d(y)
        predictions = self.predict(X, addBias=False)
        
        err = 0.5 * np.sum((predictions - y) ** 2)
        
        return err