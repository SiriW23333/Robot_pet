import numpy as np
from vision.nn.mb_tiny_RFB import Mb_Tiny_RFB
from vision.ssd.config import fd_config as config
from vision.ssd.ssd import SSD


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d."""
    return {
        'depthwise': {
            'in_channels': in_channels,
            'out_channels': in_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'groups': in_channels
        },
        'pointwise': {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': 1
        }
    }


def create_Mb_Tiny_RFB_fd(num_classes, is_test=False, device="cpu"):
    """Create an SSD model using Mb_Tiny_RFB as the base network."""
    base_net = Mb_Tiny_RFB(2)  # Initialize the base network
    base_net_model = base_net.model  # Get the base network layers

    # Define source layer indexes
    source_layer_indexes = [
        8,
        11,
        13
    ]

    # Define extra layers
    extras = [
        {
            'conv1x1': {
                'in_channels': base_net.base_channel * 16,
                'out_channels': base_net.base_channel * 4,
                'kernel_size': 1
            },
            'separable_conv': SeperableConv2d(
                in_channels=base_net.base_channel * 4,
                out_channels=base_net.base_channel * 16,
                kernel_size=3,
                stride=2,
                padding=1
            )
        }
    ]

    # Define regression headers
    regression_headers = [
        SeperableConv2d(
            in_channels=base_net.base_channel * 4,
            out_channels=3 * 4,
            kernel_size=3,
            padding=1
        ),
        SeperableConv2d(
            in_channels=base_net.base_channel * 8,
            out_channels=2 * 4,
            kernel_size=3,
            padding=1
        ),
        SeperableConv2d(
            in_channels=base_net.base_channel * 16,
            out_channels=2 * 4,
            kernel_size=3,
            padding=1
        ),
        {
            'conv': {
                'in_channels': base_net.base_channel * 16,
                'out_channels': 3 * 4,
                'kernel_size': 3,
                'padding': 1
            }
        }
    ]

    # Define classification headers
    classification_headers = [
        SeperableConv2d(
            in_channels=base_net.base_channel * 4,
            out_channels=3 * num_classes,
            kernel_size=3,
            padding=1
        ),
        SeperableConv2d(
            in_channels=base_net.base_channel * 8,
            out_channels=2 * num_classes,
            kernel_size=3,
            padding=1
        ),
        SeperableConv2d(
            in_channels=base_net.base_channel * 16,
            out_channels=2 * num_classes,
            kernel_size=3,
            padding=1
        ),
        {
            'conv': {
                'in_channels': base_net.base_channel * 16,
                'out_channels': 3 * num_classes,
                'kernel_size': 3,
                'padding': 1
            }
        }
    ]

    # Return the SSD object
    return SSD(
        num_classes=num_classes,
        base_net=base_net_model,
        source_layer_indexes=source_layer_indexes,
        extras=extras,
        classification_headers=classification_headers,
        regression_headers=regression_headers,
        is_test=is_test,
        config=config,
        device=device
    )


def create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    """Create a predictor for the SSD model."""
    class Predictor:
        def __init__(self, net, image_size, image_mean, image_std, **kwargs):
            self.net = net
            self.image_size = image_size
            self.image_mean = image_mean
            self.image_std = image_std

        def predict(self, image):
            """Simulate the prediction process."""
            confidences, locations = self.net.forward(image)
            return confidences, locations

    return Predictor(
        net=net,
        image_size=config.image_size,
        image_mean=config.image_mean_test,
        image_std=config.image_std,
        nms_method=nms_method,
        sigma=sigma
    )