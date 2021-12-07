import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    )


class PropagationNet(nn.Module):
    def __init__(self, in_dim=4, h_dim=32):
        super(PropagationNet, self).__init__()
        self.conv1 = conv_bn_relu(in_dim, h_dim)
        self.conv2 = conv_bn_relu(h_dim, h_dim)
        self.conv3 = conv_bn_relu(h_dim, h_dim)
        self.conv4 = conv_bn_relu(h_dim, 9)
        self.unfold = nn.Unfold(kernel_size=(3, 3), stride=1, padding=0)

    def forward(self, features, depth):
        x = torch.cat((features, depth), dim=1)
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        p = F.softmax(x, dim=1)

        depth_pad = F.pad(depth, (1, 1, 1, 1), mode='replicate')
        depth_unfold = self.unfold(depth_pad)

        b, c, h, w = p.shape
        p = p.view(b, c, h*w)

        out = torch.sum(p * depth_unfold, dim=1)
        out = out.view(b, h, w)
        return out