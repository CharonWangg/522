import torch
from typing import Callable


class LinearBlock(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, activation: Callable
    ) -> None:
        """
        Initialize the linear block.
        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            activation: The activation function to use
        """
        super(LinearBlock, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.activation = activation()
        self.batch_norm = torch.nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear block.
        Args:
            x: The input data.

        Returns:
            The output of the linear block.
        """
        return self.batch_norm(self.activation(self.linear(x)))


class MLP(torch.nn.Module):
    """
    A simple multi-layer perceptron.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.kaiming_normal_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.initializer = initializer
        layers = []
        for i in range(hidden_count):
            if i == 0:
                layers.append(LinearBlock(input_size, hidden_size, activation))
            else:
                layers.append(LinearBlock(hidden_size, hidden_size, activation))

        layers += [torch.nn.Dropout(0.3), torch.nn.Linear(hidden_size, num_classes)]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
