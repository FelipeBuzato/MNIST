import torch.nn as nn

class TorchModelCollection:
    def __init__(self):
        pass

    def get(self, model_name, hidden_layer_sizes):

        if(model_name == "NN"):
            return NN(hidden_layer_sizes=hidden_layer_sizes)
        
        raise ValueError(f"Unknown model: {model_name}")


class NN(nn.Module):
    def __init__(self, input_size=784, hidden_layer_sizes=(128,), output_size=10):
        super().__init__()
        
        layers = []
        sizes = [input_size] + list(hidden_layer_sizes) + [output_size]

        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)