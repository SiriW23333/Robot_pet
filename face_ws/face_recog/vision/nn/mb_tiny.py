import numpy as np
from typing import List, Tuple

class Conv2d:
    """µØÆ½Ïß¼æÈÝµÄConv2dÄ£Äâ"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        self.params = {
            'in_c': in_channels,
            'out_c': out_channels,
            'k': kernel_size,
            'stride': stride,
            'padding': padding,
            'groups': groups
        }
        
    def __call__(self, x):
        # Êµ¼Ê²¿ÊðÊ±Ó¦Ìæ»»ÎªµØÆ½ÏßBPUµÄ¾í»ýµ÷ÓÃ
        out_h = (x.shape[2] + 2*self.params['padding'] - self.params['k']) // self.params['stride'] + 1
        out_w = (x.shape[3] + 2*self.params['padding'] - self.params['k']) // self.params['stride'] + 1
        return np.random.rand(x.shape[0], self.params['out_c'], out_h, out_w).astype(np.float32)

class BatchNorm2d:
    def __call__(self, x):
        return x  # µØÆ½ÏßÄ£ÐÍÍ¨³£ÔÚÁ¿»¯Ê±ÒÑÈÚºÏBN

class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

class Sequential:
    def __init__(self, *layers):
        self.layers = layers
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = np.random.rand(out_features, in_features)
        
    def __call__(self, x):
        return x @ self.weight.T

class Mb_Tiny:
    def __init__(self, num_classes=2):
        self.base_channel = 8 * 2

        def conv_bn(inp, oup, stride):
            return Sequential(
                Conv2d(inp, oup, 3, stride, 1),
                BatchNorm2d(),
                ReLU()
            )

        def conv_dw(inp, oup, stride):
            return Sequential(
                Conv2d(inp, inp, 3, stride, 1, groups=inp),
                BatchNorm2d(),
                ReLU(),
                Conv2d(inp, oup, 1),
                BatchNorm2d(),
                ReLU()
            )

        self.model = Sequential(
            conv_bn(3, self.base_channel, 2),    # 160*120
            conv_dw(self.base_channel, self.base_channel*2, 1),
            conv_dw(self.base_channel*2, self.base_channel*2, 2),  # 80*60
            conv_dw(self.base_channel*2, self.base_channel*2, 1),
            conv_dw(self.base_channel*2, self.base_channel*4, 2),  # 40*30
            conv_dw(self.base_channel*4, self.base_channel*4, 1),
            conv_dw(self.base_channel*4, self.base_channel*4, 1),
            conv_dw(self.base_channel*4, self.base_channel*4, 1),
            conv_dw(self.base_channel*4, self.base_channel*8, 2),  # 20*15
            conv_dw(self.base_channel*8, self.base_channel*8, 1),
            conv_dw(self.base_channel*8, self.base_channel*8, 1),
            conv_dw(self.base_channel*8, self.base_channel*16, 2), # 10*8
            conv_dw(self.base_channel*16, self.base_channel*16, 1)
        )
        self.fc = Linear(1024, num_classes)

    def __call__(self, x):
        x = self.model(x)
        x = np.mean(x, axis=(2,3), keepdims=True)  # Ìæ´úavg_pool2d
        x = x.reshape(-1, 1024)                    # Ìæ´úview
        x = self.fc(x)
        return x