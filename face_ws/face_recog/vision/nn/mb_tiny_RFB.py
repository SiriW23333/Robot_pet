import numpy as np
import cv2


class BasicConv:
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.relu = relu
        self.bn = bn

    def forward(self, x):
        # Placeholder for convolution operation
        # Replace this with actual convolution logic using NumPy or OpenCV
        if self.bn:
            x = (x - np.mean(x, axis=(0, 1, 2))) / (np.std(x, axis=(0, 1, 2)) + 1e-5)  # BatchNorm simulation
        if self.relu:
            x = np.maximum(0, x)  # ReLU activation
        return x


class BasicRFB:
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        self.scale = scale
        self.out_planes = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = [
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1, dilation=vision + 1, relu=False, groups=groups)
        ]
        self.branch1 = [
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        ]
        self.branch2 = [
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        ]

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)

    def forward(self, x):
        x0 = x
        for layer in self.branch0:
            x0 = layer.forward(x0)

        x1 = x
        for layer in self.branch1:
            x1 = layer.forward(x1)

        x2 = x
        for layer in self.branch2:
            x2 = layer.forward(x2)

        out = np.concatenate((x0, x1, x2), axis=1)
        out = self.ConvLinear.forward(out)
        short = self.shortcut.forward(x)
        out = out * self.scale + short
        out = np.maximum(0, out)  # ReLU activation
        return out


class Mb_Tiny_RFB:
    def __init__(self, num_classes=2):
        self.base_channel = 8 * 2

        def conv_bn(inp, oup, stride):
            return [
                BasicConv(inp, oup, kernel_size=3, stride=stride, padding=1, relu=True, bn=True)
            ]

        def conv_dw(inp, oup, stride):
            return [
                BasicConv(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, relu=True, bn=True),
                BasicConv(inp, oup, kernel_size=1, stride=1, padding=0, relu=True, bn=True)
            ]

        self.model = [
            *conv_bn(3, self.base_channel, 2),  # 160*120
            *conv_dw(self.base_channel, self.base_channel * 2, 1),
            *conv_dw(self.base_channel * 2, self.base_channel * 2, 2),  # 80*60
            *conv_dw(self.base_channel * 2, self.base_channel * 2, 1),
            *conv_dw(self.base_channel * 2, self.base_channel * 4, 2),  # 40*30
            *conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            *conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            BasicRFB(self.base_channel * 4, self.base_channel * 4, stride=1, scale=1.0),
            *conv_dw(self.base_channel * 4, self.base_channel * 8, 2),  # 20*15
            *conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            *conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            *conv_dw(self.base_channel * 8, self.base_channel * 16, 2),  # 10*8
            *conv_dw(self.base_channel * 16, self.base_channel * 16, 1)
        ]
        self.fc_weights = np.random.rand(1024, num_classes)  # Placeholder for fully connected layer weights

    def forward(self, x):
        for layer in self.model:
            x = layer.forward(x)
        x = np.mean(x, axis=(2, 3))  # Average pooling
        x = x.reshape(-1, 1024)
        x = np.dot(x, self.fc_weights)  # Fully connected layer
        return x