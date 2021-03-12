import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from vgg16 import vgg16_layers

class AdapLayers(nn.Module):
    """Small adaptation layers.
    """

    def __init__(self, hypercolumn_layers: List[str], output_dim: int = 128):
        """Initialize one adaptation layer for every extraction point.

        Args:
            hypercolumn_layers: The list of the hypercolumn layer names.
            output_dim: The output channel dimension.
        """
        super(AdapLayers, self).__init__()
        self.layers = []
        channel_sizes = [vgg16_layers[name] for name in hypercolumn_layers]
        for i, l in enumerate(channel_sizes):
            layer = nn.Sequential(
                nn.Conv2d(l, 64, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, output_dim, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(output_dim),
            )
            self.layers.append(layer)
            self.add_module("adap_layer_{}".format(i), layer)

    def forward(self, features: List[torch.tensor]):
        """Apply adaptation layers.
        """
        for i, _ in enumerate(features):
            features[i] = getattr(self, "adap_layer_{}".format(i))(features[i])
        return features
