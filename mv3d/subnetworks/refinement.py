import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from torch_scatter import scatter


def conv1d_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs),
        torch.nn.BatchNorm1d(out_channels),
        torch.nn.ReLU(inplace=True),
    )


class HypothesisDecoder(nn.Module):
    def __init__(self, in_dim=128+128+64, h_dim=256, kernel_size=3, padding=1):
        super(HypothesisDecoder, self).__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            conv1d_bn_relu(in_dim, h_dim, kernel_size, 1, padding),
            conv1d_bn_relu(h_dim, h_dim, kernel_size, 1, padding),
            conv1d_bn_relu(h_dim, h_dim, kernel_size, 1, padding),
            nn.Conv1d(h_dim, 1, kernel_size, 1, padding)
        )
        self.sparse_interp = ME.MinkowskiInterpolation()

    def forward(self, xs, pts, pts_feat, pts_batch):
        features = pts_feat
        n_pts = pts.shape[0]
        n_hyp = pts.shape[1]
        for x in xs:
            min_pts = scatter(x['pts'], x['batch'], dim=0, reduce='min')
            pts_idx = pts - min_pts[pts_batch].unsqueeze(1).expand(*pts.shape)
            pts_idx = (pts_idx / x['res']) * x['stride']
            pts_batch_unrolled = pts_batch.unsqueeze(1).repeat(1, n_hyp).unsqueeze(2)
            pts_idx_batched = torch.cat((pts_batch_unrolled.float(), pts_idx), dim=2)
            pts_idx_batched_flat = pts_idx_batched.view(n_pts*n_hyp, 4)
            feats = self.sparse_interp(x['sparse'], pts_idx_batched_flat)
            feats = feats.view(n_pts, n_hyp, -1)
            features = feats if features is None else torch.cat((feats, features), dim=2)
        features = features.transpose(2, 1)
        preds = F.softmax(self.net(features).squeeze(1), dim=1)
        return preds

    def forward_forloop(self, xs, pts, pts_batch):
        n_batches = pts_batch.max() + 1
        n_imgs = pts.shape[0]
        n_pts_per_img = pts.shape[1]
        n_hyps = pts.shape[2]
        predictions = torch.empty((n_imgs, n_pts_per_img, n_hyps)).type_as(pts)

        for b in range(n_batches):
            # first, create dense tensors for trilinear interp
            x_dense = []
            for x in xs:
                x_batch_idx = x['batch'] == b
                f = x['feats'][x_batch_idx]
                f_idx = x['idx'][x_batch_idx]
                f_pts = x['pts'][x_batch_idx]
                f_idx = (f_idx // x['stride'])
                f_dim = f.shape[-1]

                # construct dense feature volume for trilinear interpolation with 0 padding
                f_idx += 1  # 0 padding
                shape = f_idx.max(dim=0)[0] + 2  # +2 instead of +1 for 0 padding
                x_idx, y_idx, z_idx = f_idx.transpose(1, 0)
                f_vol = torch.zeros((1, f_dim, *shape), dtype=f.dtype, device=f.device)
                f_vol[0, :, x_idx, y_idx, z_idx] = f.transpose(1, 0)
                min_pt = f_pts[0] - f_idx[0] * float(x['res'])
                scale = (shape - 1) * float(x['res'])

                x_dense.append({
                    'vol': f_vol,
                    'min_pt': min_pt,
                    'scale': scale
                })

            pts_batch_idx = torch.arange(n_imgs)[pts_batch == b]
            for b_idx in pts_batch_idx:
                pts_singleimage = pts[b_idx]
                features = None
                for x_d in x_dense:
                    q_pts = pts_singleimage - x_d['min_pt']
                    q_pts[..., 0] /= x_d['scale'][0]
                    q_pts[..., 1] /= x_d['scale'][1]
                    q_pts[..., 2] /= x_d['scale'][2]
                    grid = (float(2.) * q_pts) - float(1.)
                    grid = grid[None, None, ..., [2, 1, 0]]  # grid coordinates should be in z, y, x form!!!

                    feats = F.grid_sample(x_d['vol'], grid, 'bilinear', padding_mode='zeros', align_corners=True)
                    feats = feats[0, :, 0, ...].permute(1, 2, 0)
                    features = feats if features is None else torch.cat((feats, features), dim=2)
                features = features.transpose(2, 1)
                preds = F.softmax(self.net(features).squeeze(1), dim=1)
                predictions[b_idx] = preds
        return predictions
