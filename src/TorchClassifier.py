from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
import time
from TorchModelCollection import TorchModelCollection
import numpy as np

class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name='NN', learning_rate=1e-3, n_epochs=50, batch_size=256, sgd=True,
                       criterion='cross entropy', optimizer = 'adam', random_state=42, hidden_layer_sizes=(128,),
                       channels_conv_layers=(10,10), kernel_sizes=3, pooling_size=2, augmentation=False):
        
        # Training and optimization hyper-parameters
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.sgd = sgd
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_name = model_name
        self.random_state = random_state

        # MLP hyper-parameters
        self.hidden_layer_sizes = hidden_layer_sizes

        # CNN hyper-parameters
        self.channels_conv_layers = channels_conv_layers
        self.kernel_sizes = kernel_sizes
        self.pooling_size = pooling_size
        self.augmentation = augmentation
        self.augment_transform = v2.RandomApply([v2.RandomAffine(degrees=10, translate=(0.1, 0.1))], p=0.8)

        # Items generated only after training
        self.loss_curve_ = None
        self.validation_scores_ = None
        self.model_ = None

        self.display = True


    def _to_tensor(self, X, dtype):
        if(self.model_name == "CNN"):
            return self._to_tensor_2d(X, dtype)
        else:
            return self._to_tensor_1d(X, dtype)
        
    
    def _to_tensor_1d(self, X, dtype):
        if torch.is_tensor(X):
            return X

        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        
        return torch.as_tensor(X, dtype=dtype)
    
    
    def _to_tensor_2d(self, X, dtype):
        if(not torch.is_tensor(X)):
            if hasattr(X, "to_numpy"):
                X = X.to_numpy()
            X = torch.as_tensor(X, dtype=dtype)

        if(X.ndim == 2):
            width = height = int(np.sqrt(X.shape[-1]))
            assert width * width == X.shape[-1], "Image is not square"
            X = X.reshape(-1, 1, width, height)

        return X


    def get_model_params(self):
        if(self.model_name == "NN"):
            return {"hidden_layer_sizes": self.hidden_layer_sizes}
        
        if(self.model_name == "CNN"):
            return {"channels_conv_layers": self.channels_conv_layers,
                    "kernel_sizes": self.kernel_sizes,
                    "pooling_size": self.pooling_size,
                    "ffn_layer_sizes": self.hidden_layer_sizes}
        return {}


    def get_torch_model(self, model_name):
        params = self.get_model_params()
        torch_model_collection = TorchModelCollection()
        return torch_model_collection.get(model_name=model_name, params=params)
    

    def get_criterion(self):
        if(self.criterion == "cross entropy"):
            return nn.CrossEntropyLoss()
        
        raise ValueError(f"Criterion {self.criterion} not found.")
    

    def get_optimizer(self):
        if(self.optimizer == "adam"):
            return optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        raise ValueError(f"Optimizer {self.optimizer} not found.")
    

    def stochastic_gradient_descent(self, X, y, optimizer, criterion):
        import matplotlib.pyplot as plt
        loss_curve = []
        
        for epoch in range(self.n_epochs):
            # shuffle every epoch
            indices = torch.randperm(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
        
            epoch_loss = 0
        
            # SGD:
            for X_batch, y_batch in zip(X_shuffled.split(self.batch_size), y_shuffled.split(self.batch_size)): 
                
                # data augmentation
                if(self.augmentation):
                    X_batch = self.augment_transform(X_batch)

                # reset gradients so they won't accumulate from previous iteration
                optimizer.zero_grad(set_to_none=True)
                
                # forward propagation and loss computation
                output = self.model_(X_batch)
                loss = criterion(output, y_batch)
        
                # back propagation
                loss.backward()
                epoch_loss += loss.item() * X_batch.shape[0]
        
                # optimization step
                optimizer.step()
        
            epoch_loss /= X.shape[0]
            loss_curve.append(epoch_loss)
            
            if(epoch%20 == 0 and self.display):
                print(f"Loss after Epoch {epoch+1}: {epoch_loss:.4f}.")
        
        return loss_curve


    def full_gradient_descent(self, X, y, optimizer, criterion):
        loss_curve = []
        X_copy = X.copy()
        
        for epoch in range(self.n_epochs):    
            # data augmentation
            if(self.augmentation):
                X = self.augment_transform(X_copy)    

            optimizer.zero_grad(set_to_none=True)
            
            # forward propagation and loss computation
            output = self.model_(X)
            loss = criterion(output, y)
    
            # back propagation
            loss.backward()
            epoch_loss = loss.item()
    
            # optimization step
            optimizer.step()
        
            loss_curve.append(epoch_loss)
            
            if(epoch%20 == 0 and self.display):
                print(f"Loss after Epoch {epoch+1}: {epoch_loss:.4f}.")
        
        return loss_curve
    
    
    def fit(self, X, y):
        start_training_time = time.time()
        torch.manual_seed(self.random_state)

        # convert X and y to tensor
        X = self._to_tensor(X, dtype=torch.float32)
        y = self._to_tensor_1d(y, dtype=torch.long)

        self.model_ = self.get_torch_model(self.model_name)
        self.model_.train()
        if(self.display): print(f"Training Activated ? {self.model_.training}")
        
        # define loss function and optimizer
        criterion = self.get_criterion()
        optimizer = self.get_optimizer()

        if(self.sgd):
            loss_curve = self.stochastic_gradient_descent(X, y, optimizer=optimizer, criterion=criterion)
        else:
            loss_curve = self.full_gradient_descent(X, y, optimizer=optimizer, criterion=criterion)

        training_time = time.time() - start_training_time
        
        if(self.display): 
            print(f"Loss after gradient descent: {loss_curve[-1]:.6f}.")
            print(f"Total Training time: {training_time}.")

        self.loss_curve_ = loss_curve

        return self
    

    def predict(self, X):
        # convert X to tensor
        X = self._to_tensor(X, dtype=torch.float32)

        self.model_.eval()
        if(self.display): print(f"Training Activated ? {self.model_.training}")
    
        with torch.no_grad():
            output = self.model_(X)
            predictions = output.argmax(dim=1)
        
        return predictions.numpy()
    

    def predict_proba(self, X):
        # convert X to tensor
        X = self._to_tensor(X, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            output = self.model_(X)
        probs = torch.softmax(output, dim=1)

        return probs.numpy()