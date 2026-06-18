import numpy as np
from scipy.special import softmax
import time
from sklearn.base import BaseEstimator, ClassifierMixin

class NNClassifierFromScratch(BaseEstimator, ClassifierMixin):
    def __init__(self, activation='relu', learning_rate=5e-2, hidden_layer_sizes=(128,), n_epochs=100, batch_size=258):
        self.n_features = None
        self.n_categories = 10
        self.Z, self.A = None, None    
        self.W, self.b = None, None
        self.loss_curve_ = None
        self.layer_sizes = None
        self.n_layers = None
        self.scaling_factor = 255
        self.display_progress = False
        
        # Hyper-parameters
        self.activation = activation
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        

    def init_network(self, X, y):
        self.n_features = X.shape[1]            
        
        self.layer_sizes = (self.n_features,) + self.hidden_layer_sizes + (self.n_categories,)
        self.n_layers = len(self.layer_sizes)
        
        self.Z, self.A = {}, {}
        self.W, self.b = {}, {}
        for i in range(1, self.n_layers):
            # Weights are initialized as random numbers 
            self.W[i] = np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i])*np.sqrt(2 / self.layer_sizes[i-1])
            self.b[i] = np.zeros(self.layer_sizes[i])


    def loss_function(self, y_pred, y):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)  # to avoid infinity when applying log
        return -(y * np.log(y_pred)).sum(axis=1).mean()
        
    
    def activation_function(self, X):
        if(self.activation == 'relu'):
            return np.maximum(0, X)
        raise ValueError(f"Activation function {self.activation} not found.")


    def activation_derivative(self, X):
        if(self.activation == 'relu'):
            return (X > 0).astype(float)
        raise ValueError(f"Activation function {self.activation} not found.")
            

    def fwd_propagation(self, X):
        n_layers = self.n_layers
        self.A[0] = X.copy()
        
        for i in range(1, n_layers):
            self.Z[i] = self.A[i-1] @ self.W[i] + self.b[i]
            if(i != n_layers-1):
                self.A[i] = self.activation_function(self.Z[i])
            else:
                self.A[i] = softmax(self.Z[i], axis=1)


    def back_propagation(self, y):
        n_layers = self.n_layers
        
        for i in range(n_layers-1, 0, -1):
            if(i == n_layers-1):
                # Last layer derivatives
                dZ = self.A[i] - y
            else:
                # Hidden layer derivatives
                dZ = (dZ @ self.W[i+1].T) * self.activation_derivative(self.Z[i])

            # Gradients of layer i
            n_observations = y.shape[0]
            dW = (1/n_observations) * self.A[i-1].T @ dZ
            dB = (1/n_observations) * np.sum(dZ, axis=0)

            # update weights of layer i
            self.W[i] -= self.learning_rate * dW
            self.b[i] -= self.learning_rate * dB


    def full_gradient_descent(self, X, y, n_epochs, display_progress=False):
        n_layers = self.n_layers
        loss_curve = []
        start = time.time()
        
        for i in range(n_epochs):
            if(display_progress and i%20 == 0): print(f"Starting Epoch {i+1}...")
                
            self.fwd_propagation(X)
            loss = self.loss_function(self.A[n_layers-1], y)
            loss_curve.append(loss)
            self.back_propagation(y)
            
            if(display_progress and i%20 == 0): print(f"Epoch {i+1} done. Loss = {loss}.")

        # Loss at the end of gradient descent
        self.fwd_propagation(X)
        loss = self.loss_function(self.A[n_layers-1], y)
        loss_curve.append(loss)
        self.loss_curve_ = loss_curve
        time_taken = time.time() - start
        if(display_progress): print(f"Loss after gradient descent = {loss}.\nTime taken: {time_taken}")


    def stochastic_gradient_descent(self, X, y, n_epochs, display_progress=False):
        n_layers = self.n_layers
        loss_curve = []
        start_time = time.time()
        
        for i in range(n_epochs):
            if(display_progress and i%20 == 0): print(f"Starting Epoch {i+1}...")

            # split dataset in bacthes
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            loss_curve_batch = []
            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                self.fwd_propagation(X_batch)
                loss = self.loss_function(self.A[n_layers-1], y_batch)
                loss_curve_batch.append(loss)
                self.back_propagation(y_batch)
            
            loss = np.mean(loss_curve_batch)
            loss_curve.append(loss)
            if(display_progress and i%20 == 0): print(f"Epoch {i+1} done. Loss = {loss}.")

        # Loss at the end of gradient descent
        self.fwd_propagation(X)
        loss = self.loss_function(self.A[n_layers-1], y)
        loss_curve.append(loss)
        self.loss_curve_ = loss_curve
        time_taken = time.time() - start_time
        if(display_progress): print(f"Loss after gradient descent = {loss}.\nTime taken: {time_taken}")


    def fit(self, X, y):
        # one hot encode y and transform X to numpy matrix
        y_onehot = np.eye(self.n_categories)[np.asarray(y)]
        X_np = np.asarray(X.copy()) / self.scaling_factor

        # Initialize the network
        self.init_network(X_np, y_onehot)
        
        # call gradient descent
        if(self.batch_size is None):
            self.full_gradient_descent(X_np, y_onehot, self.n_epochs, self.display_progress)
        else:
            self.stochastic_gradient_descent(X_np, y_onehot, self.n_epochs, self.display_progress)
        
        # Just to make it Sklearn compatible (we need at least one atribute ending with _)
        self.n_features_in_ = X_np.shape[1]
        
        return self


    def predict(self, X):
        X_np = np.asarray(X.copy()) / self.scaling_factor
        
        # call forward propagation
        self.fwd_propagation(X_np)
        
        # return the category with highest propability
        A = self.A[self.n_layers-1].copy()
        y_pred = np.argmax(A, axis=1)
        
        return y_pred


    def predict_proba(self, X):
        X = np.asarray(X) / self.scaling_factor
        self.fwd_propagation(X)
        return self.A[self.n_layers - 1]

