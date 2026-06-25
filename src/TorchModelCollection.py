import torch.nn as nn

class TorchModelCollection:
    def __init__(self):
        pass

    def get(self, model_name, params):

        if(model_name == "NN"):
            return NN(**params)
        
        elif(model_name == "CNN"):
            return CNN(**params)
        
        raise ValueError(f"Unknown model: {model_name}")


class CNN(nn.Module):
    def __init__(self, channels_conv_layers=(10,10), kernel_sizes=3, pooling_size=2, ffn_layer_sizes=(128,), output_size=10):
        super().__init__()
        
        layers = []

        # build convolutional layers
        channels_conv_layers = (1,) + channels_conv_layers
        n_conv_layers = len(channels_conv_layers)

        for i in range(1, n_conv_layers):
            layers.append(nn.Conv2d(in_channels=channels_conv_layers[i-1], out_channels=channels_conv_layers[i], 
                                    kernel_size=kernel_sizes, padding='valid', stride=1))
            layers.append(nn.BatchNorm2d(num_features=channels_conv_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size))

        # Flatten last convolutional output
        layers.append(nn.Flatten(start_dim=1))

        # build feed forward layers
        layers.append(nn.LazyLinear(ffn_layer_sizes[0]))
        layers.append(nn.ReLU())
        ffn_layer_sizes = list(ffn_layer_sizes) + [output_size]

        for i in range(len(ffn_layer_sizes)-1):
            layers.append(nn.Linear(ffn_layer_sizes[i], ffn_layer_sizes[i+1]))
            if i < len(ffn_layer_sizes)-2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)


    def forward(self, X):
        return self.network(X)


class NN(nn.Module):
    def __init__(self, hidden_layer_sizes=(128,), output_size=10):
        super().__init__()
        
        sizes = list(hidden_layer_sizes) + [output_size]
        layers = []
        layers.append(nn.LazyLinear(hidden_layer_sizes[0]))
        layers.append(nn.ReLU())

        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)


    def forward(self, X):
        return self.network(X)