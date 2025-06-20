import torch
import torch.nn as nn

class GenomeMLP(nn.Module):
    """
    A simple fully connected neural network (MLP) for genome inference.
    Input: binary genome read (as float tensor)
    Output: probability estimates for each base being 1
    """
    def __init__(self, input_dim):
        super(GenomeMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), # Hidden layer with 64 neurons
            nn.ReLU(),                # Activation function
            nn.Linear(64, input_dim),  # Output same length as input
            nn.Sigmoid()  # To output probabilities between 0 and 1
        )

    def forward(self, x):
        return self.model(x)

        