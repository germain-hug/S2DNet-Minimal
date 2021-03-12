import torch
import torch.nn as nn

from torchvision import models
from typing import List, Dict

from adap_layers import AdapLayers
from vgg16 import vgg16_layers


class S2DNet(nn.Module):
    """The S2DNet model
    """

    def __init__(
        self,
        device: torch.device,
        hypercolumn_layers: List[str] = ["conv1_2", "conv3_3", "conv5_3"],
        checkpoint_path: str = None,
    ):
        """Initialize S2DNet.

        Args:
            device: The torch device to put the model on
            hypercolumn_layers: Names of the layers to extract features from
            checkpoint_path: Path to the pre-trained model.
        """
        super(S2DNet, self).__init__()
        self._device = device
        self._checkpoint_path = checkpoint_path
        self.layer_to_index = dict((k, v) for v, k in enumerate(vgg16_layers.keys()))
        self._hypercolumn_layers = hypercolumn_layers

        # Initialize architecture
        vgg16 = models.vgg16(pretrained=False)
        layers = list(vgg16.features.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.adaptation_layers = AdapLayers(self._hypercolumn_layers).to(self._device)
        self.eval()

        # Restore params from checkpoint
        if checkpoint_path:
            print(">> Loading weights from {}".format(checkpoint_path))
            self._checkpoint = torch.load(checkpoint_path, map_location=device)
            self._hypercolumn_layers = self._checkpoint["hypercolumn_layers"]
            self.load_state_dict(self._checkpoint["state_dict"])

    def forward(self, image_tensor: torch.FloatTensor):
        """Compute intermediate feature maps at the provided extraction levels.

        Args:
            image_tensor: The [N x 3 x H x Ws] input image tensor.
        Returns:
            feature_maps: The list of output feature maps.
        """
        feature_maps, j = [], 0
        feature_map = image_tensor
        layer_list = list(self.encoder.modules())[0]
        for i, layer in enumerate(layer_list):
            feature_map = layer(feature_map)
            if j < len(self._hypercolumn_layers):
                next_extraction_index = self.layer_to_index[self._hypercolumn_layers[j]]
                if i == next_extraction_index:
                    feature_maps.append(feature_map)
                    j += 1
        feature_maps = self.adaptation_layers(feature_maps)
        return feature_maps
