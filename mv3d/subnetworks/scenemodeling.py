import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from torch_scatter import scatter


def sparse_conv_bn_relu3d(in_channels, out_channels):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=1, dimension=3, bias=False),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True)
    )


class SparseResidual3d(nn.Module):
    def __init__(self, feat_dim, norm="gn", num_groups=None):
        super(SparseResidual3d, self).__init__()
        has_bias = norm is None
        if norm == "gn":
            self.n1 = MinkowskiGroupNorm(num_groups, feat_dim)
            self.n2 = MinkowskiGroupNorm(num_groups, feat_dim)

            # initialization
            nn.init.constant_(self.n2.gn.weight, 0)
        elif norm == "bn":
            self.n1 = ME.MinkowskiBatchNorm(feat_dim)
            self.n2 = ME.MinkowskiBatchNorm(feat_dim)

            # initialization
            nn.init.constant_(self.n2.bn.weight, 0)
        else:
            self.n1 = lambda x: x
            self.n2 = lambda x: x

        self.conv1 = ME.MinkowskiConvolution(feat_dim, feat_dim, kernel_size=3, stride=1, dimension=3, bias=has_bias)
        self.relu1 = ME.MinkowskiReLU(inplace=True)
        self.conv2 = ME.MinkowskiConvolution(feat_dim, feat_dim, kernel_size=3, stride=1, dimension=3, bias=has_bias)
        self.relu2 = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.n2(self.conv2(self.relu1(self.n1(self.conv1(x)))))
        out += x
        return self.relu2(out)


class Residual1D(nn.Module):
    # copied and modified slightly from https://github.com/autonomousvision/convolutional_occupancy_networks/
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        self.shortcut = nn.Linear(size_in, size_out, bias=False) if size_in != size_out else lambda x: x

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        x_s = self.shortcut(x)
        return x_s + dx


class MinkowskiGroupNorm(nn.Module):
    """A group normalization layer for a sparse tensor.
    See the pytorch :attr:`torch.nn.GroupNorm` for more details.
    """

    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-5,
        affine=True
    ):
        super(MinkowskiGroupNorm, self).__init__()
        self.gn = torch.nn.GroupNorm(
            num_groups,
            num_channels,
            eps=eps,
            affine=affine
        )

    def forward(self, input):
        output = self.gn(input.F)
        return ME.SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )

    def __repr__(self):
        s = "({}, {}, eps={}, affine={})".format(
            self.gn.num_groups,
            self.gn.num_channels,
            self.gn.eps,
            self.gn.affine,
        )
        return self.__class__.__name__ + s


class PointNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, in_dim=3):
        super(PointNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_pos = nn.Linear(in_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, pts, idx, n_idx):
        x = self.fc1(F.relu(self.fc_pos(pts)))
        x_pool = scatter(x, idx, dim=0, reduce='max', dim_size=n_idx)
        x = torch.cat((x, x_pool[idx]), dim=1)

        x = self.fc2(F.relu(x))
        x_pool = scatter(x, idx, dim=0, reduce='max', dim_size=n_idx)
        x = torch.cat((x, x_pool[idx]), dim=1)

        x = self.fc3(F.relu(x))
        x_pool = scatter(x, idx, dim=0, reduce='max', dim_size=n_idx)
        x = torch.cat((x, x_pool[idx]), dim=1)

        x = self.fc4(F.relu(x))
        x_pool = scatter(x, idx, dim=0, reduce='max', dim_size=n_idx)
        x = self.fc_out(F.relu(x_pool))

        return x


class SparseUNet(nn.Module):
    def __init__(self, dims=(64, 128, 128), n_groups=(4, 8, 8), n_res=(1, 2, 3)):
        super(SparseUNet, self).__init__()
        self.res_down = nn.ModuleList()
        for i, n in enumerate(n_res):
            layers = []
            for l in range(n):
                layers.append(SparseResidual3d(dims[i], norm="gn", num_groups=n_groups[i]))
            self.res_down.append(nn.Sequential(*layers))

        self.down = nn.ModuleList()
        for i in range(1, len(dims)):
            self.down.append(nn.Sequential(
                ME.MinkowskiConvolution(dims[i-1], dims[i], kernel_size=3, stride=2, dimension=3),
                MinkowskiGroupNorm(n_groups[i], dims[i]),
                ME.MinkowskiReLU(inplace=True)
            ))

        # reverse lists
        n_res = n_res[::-1]
        dims = dims[::-1]
        n_groups = n_groups[::-1]

        self.res_up = nn.ModuleList()
        for i, n in enumerate(n_res[1:]):
            layers = []
            for l in range(n):
                layers.append(SparseResidual3d(dims[i+1], norm="gn", num_groups=n_groups[i+1]))
            self.res_up.append(nn.Sequential(*layers))

        self.up = nn.ModuleList()
        self.feat_adj = nn.ModuleList()
        for i in range(1, len(dims)):
            self.up.append(nn.Sequential(
                ME.MinkowskiConvolutionTranspose(dims[i-1], dims[i], kernel_size=3, stride=2, dimension=3),
                MinkowskiGroupNorm(n_groups[i], dims[i]),
                ME.MinkowskiReLU(inplace=True)
            ))
            self.feat_adj.append(nn.Sequential(
                ME.MinkowskiConvolution(2*dims[i], dims[i], kernel_size=1, stride=1, dimension=3),
                MinkowskiGroupNorm(n_groups[i], dims[i]),
                ME.MinkowskiReLU(inplace=True)
            ))

    def forward(self, F, pts, idx, batch, res):
        idx_batched = torch.cat((batch.unsqueeze(1), idx), dim=1).int().to(F.device)  # [Nx4] batched idx tensor

        x = ME.SparseTensor(F, idx_batched)
        x = self.res_down[0](x)
        xs = [x, ]

        for i, layer in enumerate(self.res_down[1:]):
            x = layer(self.down[i](x))
            xs.append(x)

        xs = xs[::-1]   # reverse list

        out = [xs[0]]
        for i, layer in enumerate(self.res_up):
            x = self.feat_adj[i](ME.cat(self.up[i](x), xs[i + 1]))  # transpose, concat, feature adjust
            x = layer(x)    # residuals
            out.append(x)

        out_info = []
        for x in out:
            # determine new anchor point locations
            x_idx = x.C[:, 1:].type_as(batch)
            x_batch = x.C[:, 0].type_as(batch)
            x_feat = x.F
            x_stride = x.tensor_stride[0]
            x_res = x_stride * res
            # determine spatial locations of new cell features
            with torch.no_grad():
                x_pts = torch.empty((x_batch.shape[0], 3), dtype=torch.float, device=pts.device)
                n_batches = torch.max(batch).item() + 1
                for i in range(n_batches):
                    batch_idx_in = batch == i
                    batch_idx_out = x_batch == i
                    pts_min = pts[batch_idx_in][0] - (idx[batch_idx_in][0] * res)
                    x_pts[batch_idx_out] = x_idx[batch_idx_out] * res + pts_min
            out_info.append({
                'feats': x_feat,
                'pts': x_pts,
                'res': x_res,
                'batch': x_batch,
                'idx': x_idx,
                'stride': x.tensor_stride[0],
                'sparse': x
            })

        return out_info