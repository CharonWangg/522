import torch
from typing import Callable


class LinearBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, activation):
        """
        Initialize the linear block.
        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            activation: The activation function to use.
        """
        super(LinearBlock, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.activation = activation()
        self.batch_norm = torch.nn.BatchNorm1d(out_features)

    def forward(self, x):
        """
        Forward pass of the linear block.
        Args:
            x: The input data.

        Returns:
            The output of the linear block.
        """
        return self.batch_norm(self.activation(self.linear(x)))


class MLP(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: list,
            num_classes: int,
            activation: Callable = torch.nn.ReLU,
            initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The list of number of neurons H in the each hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.initializer = initializer
        layers = []
        for i in range(len(hidden_size)):
            if i == 0:
                layers.append(LinearBlock(input_size, hidden_size[i], activation))
            else:
                layers.append(LinearBlock(hidden_size[i - 1], hidden_size[i], activation))

        layers += [torch.nn.Dropout(0.3), torch.nn.Linear(hidden_size[-1], num_classes)]
        self.layers = torch.nn.Sequential(*layers)
        self._initialize_weights(initializer)

    def _initialize_weights(self, initializer: Callable) -> None:
        """
        Initialize the weights of the network.

        Arguments:
            initializer: The initializer to use for the weights.
        """
        for m in self.layers.modules():
            if isinstance(m, torch.nn.Linear):
                initializer(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # flatten image into a vector
        x = x.view(x.size(0), -1)
        # pass through the network
        return self.layers(x)
