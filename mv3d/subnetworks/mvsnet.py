import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_scatter import scatter
from mv3d import utils
from collections import OrderedDict


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    )


class ConvBnRelu3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnRelu3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DeconvBnRelu3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, **kwargs):
        super(DeconvBnRelu3d, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                         bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.deconv(x)), inplace=True)


class Residual3d(nn.Module):
    def __init__(self, feat_dim, kernel_size=3, padding=1):
        super(Residual3d, self).__init__()
        self.conv1 = ConvBnRelu3d(feat_dim, feat_dim, kernel_size, 1, padding)
        self.conv2 = nn.Conv3d(feat_dim, feat_dim, kernel_size, 1, padding, bias=False)
        self.bn = nn.BatchNorm3d(feat_dim)

        # initialization
        nn.init.constant_(self.bn.weight, 0)

    def forward(self, x):
        out = self.bn(self.conv2(self.conv1(x)))
        out += x
        return F.relu(x)


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        backbone_mobile_layers = list(torchvision.models.mnasnet1_0(pretrained=True).layers.children())

        self.layer1 = torch.nn.Sequential(*backbone_mobile_layers[0:8])
        self.layer2 = torch.nn.Sequential(*backbone_mobile_layers[8:9])
        self.layer3 = torch.nn.Sequential(*backbone_mobile_layers[9:10])
        self.layer4 = torch.nn.Sequential(*backbone_mobile_layers[10:12])
        self.layer5 = torch.nn.Sequential(*backbone_mobile_layers[12:14])

    def forward(self, image):
        layer1 = self.layer1(image)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        return layer1, layer2, layer3, layer4, layer5

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(FeatureExtractor, self).train(mode)
        self.apply(utils.freeze_batchnorm)


class FeatureShrinker(torch.nn.Module):
    def __init__(self, feat_dim):
        super(FeatureShrinker, self).__init__()
        self.fpn = torchvision.ops.FeaturePyramidNetwork(in_channels_list=[16, 24, 40, 96, 320],
                                                         out_channels=feat_dim,
                                                         extra_blocks=None)

    def forward(self, layer1, layer2, layer3, layer4, layer5):
        fpn_input = OrderedDict()
        fpn_input['layer1'] = layer1
        fpn_input['layer2'] = layer2
        fpn_input['layer3'] = layer3
        fpn_input['layer4'] = layer4
        fpn_input['layer5'] = layer5
        fpn_output = self.fpn(fpn_input)

        features_half = fpn_output['layer1']
        features_quarter = fpn_output['layer2']
        features_eighth = fpn_output['layer3']
        features_sixteenth = fpn_output['layer4']
        features_thirtysecond = fpn_output['layer5']

        return features_half, features_quarter, features_eighth, features_sixteenth, features_thirtysecond


class FeatureAggregator(torch.nn.Module):
    def __init__(self, in_channels=64, out_channels=32):
        super().__init__()
        self.scale_thirtysecond = torch.nn.Sequential(
            conv_bn_relu(in_channels, in_channels),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            conv_bn_relu(in_channels, out_channels),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.scale_sixteenth = torch.nn.Sequential(
            conv_bn_relu(in_channels, out_channels),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.scale_eighth = conv_bn_relu(in_channels, out_channels)

    def forward(self, features_half, features_quarter, features_eighth, features_sixteenth, features_thirtysecond):
        features_thirtysecond_to_eighth = self.scale_thirtysecond(features_thirtysecond)
        features_sixteenth_to_eighth = self.scale_sixteenth(features_sixteenth)
        features_eighth_to_eighth = self.scale_eighth(features_eighth)
        consolidated_features = features_thirtysecond_to_eighth + \
                                features_sixteenth_to_eighth + \
                                features_eighth_to_eighth
        return consolidated_features


class CostRegNet(nn.Module):
    """Copied and modified slightly from https://github.com/xy-guo/MVSNet_pytorch"""
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnRelu3d(in_channels, base_channels)

        self.conv1 = ConvBnRelu3d(base_channels, 2*base_channels, stride=2)
        self.conv2 = ConvBnRelu3d(2*base_channels, 2*base_channels)

        self.conv3 = ConvBnRelu3d(2*base_channels, 4*base_channels, stride=2)
        self.conv4 = ConvBnRelu3d(4*base_channels, 4*base_channels)

        self.conv5 = ConvBnRelu3d(4*base_channels, 8*base_channels, stride=2)
        self.conv6 = ConvBnRelu3d(8*base_channels, 8*base_channels)

        self.conv7 = DeconvBnRelu3d(8*base_channels, 4*base_channels, output_padding=1)
        self.conv8 = DeconvBnRelu3d(4*base_channels, 2*base_channels)
        self.conv9 = DeconvBnRelu3d(2*base_channels, base_channels, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv8(x)
        x = conv0 + self.conv9(x)
        x = self.prob(x)
        return x


class MVSNet(nn.Module):
    def __init__(self, feat_dim=32, img_size=(240, 320)):
        super(MVSNet, self).__init__()
        self.feat_dim = feat_dim
        self.img_size = img_size

        self.feat_extractor = FeatureExtractor()
        self.feat_shrinker = FeatureShrinker(feat_dim)
        self.cnn_3d = CostRegNet(feat_dim, 8)

    def forward(self, batch, depth_start, depth_interval, n_planes, depth_img_size):
        depth_end = depth_start + depth_interval*(n_planes-1)
        n_imgs, n_channel, img_height, img_width = batch.images.shape
        ref_idx, gather_idx = torch.unique(batch.ref_src_edges[0], return_inverse=True)
        n_ref_imgs = len(ref_idx)

        # feature extraction
        features_half, features_quarter, features_eighth, features_sixteenth, features_thirtysecond = \
            self.feat_shrinker(*self.feat_extractor(batch.images))

        # feature warping
        with torch.no_grad():
            pts = utils.batched_build_plane_sweep_volume_tensor(depth_start, depth_interval, n_planes, batch.rotmats,
                                                                batch.tvecs, batch.K, self.img_size, depth_img_size)
            pts = pts.type_as(batch.images)
            n_pts = pts.shape[2]
            w = torch.ones((n_imgs, 1, n_pts), dtype=torch.float32).type_as(batch.images)
            pts_H = torch.cat((pts, w), dim=1)

            # building projection matrix
            P = torch.cat((batch.rotmats, batch.tvecs[..., None]), dim=2)
            P = torch.bmm(batch.K, P)

            pts_img_H = torch.bmm(P[batch.ref_src_edges[1]], pts_H[batch.ref_src_edges[0]])
            z_buffer = pts_img_H[:, 2]
            z_buffer = torch.abs(z_buffer) + 1e-8  # add safety to ensure no div/0
            pts_img = pts_img_H[:, :2] / z_buffer[:, None]

            grid = pts_img.transpose(2, 1).view(pts_img.shape[0], n_pts, 1, 2)
            grid[..., 0] = (grid[..., 0] / float(self.img_size[1] - 1)) * 2 - 1.0  # normalize to [-1, 1]
            grid[..., 1] = (grid[..., 1] / float(self.img_size[0] - 1)) * 2 - 1.0  # normalize to [-1, 1]

        # feature fetching
        x_vox = F.grid_sample(features_quarter[batch.ref_src_edges[1]], grid, mode='bilinear',
                              align_corners=True)
        x_vox = x_vox.squeeze(3).view(-1, self.feat_dim, n_planes, *depth_img_size)

        # variance aggregation
        x_avg = scatter(x_vox, gather_idx, dim=0, reduce='mean', dim_size=n_ref_imgs)
        x_avg_sq = scatter(x_vox**2, gather_idx, dim=0, reduce='mean', dim_size=n_ref_imgs)
        x_var = x_avg_sq - x_avg**2

        # cost volume regularization
        x_reg = self.cnn_3d(x_var).squeeze(1)
        x_prob = F.softmax(-x_reg, dim=1)

        # coarse depth prediction using expectation
        depth_vals = torch.linspace(depth_start, depth_end, n_planes).type_as(batch.images)
        depth_volume = depth_vals.unsqueeze(0).repeat(n_ref_imgs, 1)
        depth_volume = depth_volume.view(n_ref_imgs, n_planes, 1, 1).expand(x_prob.shape)

        depth_img = torch.sum(depth_volume * x_prob, dim=1)  # B x dW x dH

        return depth_img, features_half, features_quarter, features_eighth


if __name__ == '__main__':
    net1 = FeatureExtractor()
    net2 = FeatureShrinker(64)
    net3 = FeatureAggregator(64, 32)
    x = torch.rand(10, 3, 256, 320)
    out = net2(*net1(x))
    out = net3(*out)
    print(out.shape)
