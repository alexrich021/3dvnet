import pytorch_lightning as pl
from mv3d.subnetworks.mvsnet import MVSNet
from mv3d.subnetworks.scenemodeling import PointNet, SparseUNet
from mv3d.subnetworks.refinement import HypothesisDecoder
from mv3d.subnetworks.upsampling import PropagationNet
from mv3d.loss import MAELoss
from mv3d.eval.metricfunctions import calc_2d_depth_metrics
import torch
import torch.nn.functional as F
from mv3d import utils
from torch_scatter import scatter


class PL3DVNet(pl.LightningModule):
    """
    Pytorch Lightning implementation of 3DVNet
    """
    def __init__(self, depth_train, depth_test, edge_len, feat_dim=16, img_size=(256, 320), hyp_ksize=3, hyp_pad=1,
                 lr=1e-3, lr_step=100, lr_gamma=0.1, finetune=False):
        super().__init__()
        # hparams
        self.depth_train = depth_train
        self.depth_test = depth_test
        self.edge_len = edge_len
        self.feat_dim = feat_dim
        self.img_size = img_size
        self.hyp_ksize = hyp_ksize
        self.hyp_pad = hyp_pad
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.finetune = finetune
        self.save_hyperparameters()

        # networks
        self.mvsnet = MVSNet(self.hparams.feat_dim, self.hparams.img_size)
        self.pointnet = PointNet(4*self.hparams.feat_dim, 2*self.hparams.feat_dim, self.hparams.feat_dim+3)
        self.sparse_conv = SparseUNet(dims=(2*self.hparams.feat_dim, 128, 128), n_groups=(4, 8, 8), n_res=(1, 2, 3))
        self.decoder = HypothesisDecoder(128+128+3*self.hparams.feat_dim, 128, self.hparams.hyp_ksize,
                                         self.hparams.hyp_pad)
        self.refine_quarter = PropagationNet(in_dim=self.hparams.feat_dim+1, h_dim=32)
        self.refine_half = PropagationNet(in_dim=self.hparams.feat_dim+1, h_dim=32)
        self.refine_full = PropagationNet(in_dim=3+1, h_dim=32)

        # losses
        self.mae_loss = MAELoss()

    def forward(self, batch, offsets, n_iters):
        loss = torch.tensor(0., dtype=torch.float32, device=self.device, requires_grad=False)
        out = {'ref': []}   # output dictionary to be returned

        # make initial depth predictions
        depth_config = self.hparams.depth_train if self.training else self.hparams.depth_test
        depth_pred, depth_batch, feats_half, feats_quarter, feats_eighth, ref_idx = \
            self.make_initial_depth_predictions(batch, depth_config)

        # initial depth supervision
        depth_gt_sm = F.interpolate(batch.depth_images.unsqueeze(1), depth_pred.shape[-2:], mode='nearest').squeeze(1)
        loss_2d = self.mae_loss(depth_pred, depth_gt_sm, self.hparams.depth_test['depth_interval'])
        metrics_2d = calc_2d_depth_metrics(depth_pred, depth_gt_sm)
        out['initial'] = metrics_2d
        out['loss_2d'] = loss_2d
        loss += loss_2d

        _lambda = 1. if self.hparams.finetune else min(self.current_epoch, 10) * 0.1

        for i in range(n_iters):
            xs = self.model_scene(depth_pred, depth_batch, feats_quarter, batch.rotmats, batch.tvecs, batch.K,
                                  batch.ref_src_edges)
            for j, offset in enumerate(offsets):
                offset_pred = self.run_pointflow(xs, depth_pred, depth_batch, feats_quarter, batch.rotmats, batch.tvecs,
                                                 batch.K, batch.ref_src_edges, offset, 3)
                depth_pred += offset_pred
                metrics = calc_2d_depth_metrics(depth_pred, depth_gt_sm)

                # supervise all intermediate predictions
                loss_offset = self.mae_loss(depth_pred, depth_gt_sm, self.hparams.depth_test['depth_interval'])
                metrics['loss_2d'] = loss_offset
                loss += _lambda * loss_offset

                out['ref'].append(metrics)

        # starting size --> 1/4
        depth_pred = F.interpolate(depth_pred.unsqueeze(1), feats_quarter.shape[-2:], mode='nearest').squeeze(1)
        depth_pred = self.refine_quarter(feats_quarter[ref_idx], depth_pred.unsqueeze(1)).squeeze(1)

        # 1/4 supervision
        depth_gt_quarter = F.interpolate(batch.depth_images.unsqueeze(1), depth_pred.shape[-2:], mode='nearest').squeeze(1)
        loss_refine_quarter = self.mae_loss(depth_pred, depth_gt_quarter, self.hparams.depth_test['depth_interval'])
        metrics = calc_2d_depth_metrics(depth_pred, depth_gt_quarter)
        metrics['loss_2d'] = loss_refine_quarter
        out['quarter'] = metrics

        # 1/4 --> 1/2
        depth_pred = F.interpolate(depth_pred.unsqueeze(1), feats_half.shape[-2:], mode='nearest').squeeze(1)
        depth_pred = self.refine_half(feats_half[ref_idx], depth_pred.unsqueeze(1)).squeeze(1)

        # 1/2 supervision
        depth_gt_half = F.interpolate(batch.depth_images.unsqueeze(1), depth_pred.shape[-2:], mode='nearest').squeeze(1)
        loss_refine_half = self.mae_loss(depth_pred, depth_gt_half, self.hparams.depth_test['depth_interval'])
        metrics = calc_2d_depth_metrics(depth_pred, depth_gt_half)
        metrics['loss_2d'] = loss_refine_half
        out['half'] = metrics

        # 1/2 --> image size
        depth_pred = F.interpolate(depth_pred.unsqueeze(1), batch.images.shape[-2:], mode='nearest').squeeze(1)
        depth_pred = self.refine_full(batch.images[ref_idx], depth_pred.unsqueeze(1)).squeeze(1)

        # final supervision
        loss_refine = self.mae_loss(depth_pred, batch.depth_images, self.hparams.depth_test['depth_interval'])
        metrics = calc_2d_depth_metrics(depth_pred, batch.depth_images)
        metrics['loss_2d'] = loss_refine
        out['final'] = metrics

        out['loss'] = loss
        return out

    def make_initial_depth_predictions(self, batch, depth_config):
        depth_pred, feats_half, feats_quarter, features_eighth = self.mvsnet(
            batch, depth_config['depth_start'], depth_config['depth_interval'], depth_config['n_intervals'],
            depth_config['size'])
        ref_idx = torch.unique(batch.ref_src_edges[0])
        depth_batch = batch.images_batch[ref_idx]
        return depth_pred, depth_batch, feats_half, feats_quarter, features_eighth, ref_idx

    def construct_feature_rich_pointcloud(self, depth_pred, depth_batch, img_feats, rotmats, tvecs, K, ref_src_edges):
        ref_idx, gather_idx = torch.unique(ref_src_edges[0], return_inverse=True)
        n_imgs = depth_pred.shape[0]

        # back-project depth points to world coordinates
        with torch.no_grad():
            K_inv = torch.inverse(K[ref_idx])
            R_T = rotmats[ref_idx].transpose(2, 1)
            pts_img = utils.batched_build_img_pts_tensor(n_imgs, self.hparams.img_size, depth_pred.shape[1:])
            pts_img = pts_img.type_as(depth_pred)
        depth_flat = depth_pred.view(n_imgs, 1, -1)
        pts_img = pts_img * depth_flat
        pts = torch.bmm(R_T, torch.bmm(K_inv, pts_img) - tvecs[ref_idx].unsqueeze(-1))

        # re-project depth points to all source images and generate a feature for each point
        with torch.no_grad():
            n_pts = pts.shape[2]
            w = torch.ones((n_imgs, 1, n_pts), dtype=torch.float32).type_as(pts)
            pts_H = torch.cat((pts, w), dim=1)

            # building projection matrix
            P = torch.cat((rotmats, tvecs[..., None]), dim=2)
            P = torch.bmm(K, P)

            pts_img_H = torch.bmm(P[ref_src_edges[1]], pts_H[gather_idx])
            z_buffer = pts_img_H[:, 2]
            z_buffer = torch.abs(z_buffer) + 1e-8  # add safety to ensure no div/0
            pts_img = pts_img_H[:, :2] / z_buffer[:, None]

            grid = pts_img.transpose(2, 1).view(pts_img.shape[0], n_pts, 1, 2)
            grid[..., 0] = (grid[..., 0] / float(self.hparams.img_size[1] - 1)) * 2 - 1.0  # normalize to [-1, 1]
            grid[..., 1] = (grid[..., 1] / float(self.hparams.img_size[0] - 1)) * 2 - 1.0  # normalize to [-1, 1]

        x = F.grid_sample(img_feats[ref_src_edges[1]], grid, mode='bilinear', align_corners=True)
        x = x.squeeze(3)
        x_avg = scatter(x, gather_idx, dim=0, reduce='mean')
        x_avg_sq = scatter(x ** 2, gather_idx, dim=0, reduce='mean')
        x_var = x_avg_sq - x_avg ** 2

        pts = pts.transpose(2, 1).reshape(-1, 3)
        pts_feat = x_var.transpose(2, 1).reshape(-1, self.hparams.feat_dim)
        pts_batch = depth_batch.unsqueeze(1).expand(n_imgs, depth_pred.shape[1] * depth_pred.shape[2]).reshape(-1)
        return pts, pts_feat, pts_batch

    def model_scene(self, depth_pred, depth_batch, img_feats, rotmats, tvecs, K, ref_src_edges, return_pts=False):
        pts, pts_feat, pts_batch = self.construct_feature_rich_pointcloud(depth_pred, depth_batch, img_feats, rotmats,
                                                                          tvecs, K, ref_src_edges)

        anchor_pts, anchor_idx3d, anchor_batch, anchor_pts_edges = utils.voxelize(pts, pts_batch, self.edge_len)
        n_anchors = anchor_pts.shape[0]
        x = torch.cat((pts[anchor_pts_edges[1]]-anchor_pts[anchor_pts_edges[0]], pts_feat[anchor_pts_edges[1]]), dim=1)
        x = self.pointnet(x, anchor_pts_edges[0], n_anchors)
        xs = self.sparse_conv(x, anchor_pts, anchor_idx3d, anchor_batch, self.edge_len)
        return (xs, pts) if return_pts else xs

    def run_pointflow(self, xs, depth_pred, depth_batch, img_feats, rotmats, tvecs, K, ref_src_edges, offset, n):
        n_imgs = depth_pred.shape[0]
        ref_idx, gather_idx = torch.unique(ref_src_edges[0], return_inverse=True)

        with torch.no_grad():
            # FIRST, we back-project our depth pts to determine our point hypotheses
            K_inv = torch.inverse(K[ref_idx])
            R_T = rotmats[ref_idx].transpose(2, 1)
            pts_img = utils.batched_build_img_pts_tensor(n_imgs, self.hparams.img_size, depth_pred.shape[1:])
            pts_img = pts_img.type_as(depth_pred)
            pts_batch = depth_batch.unsqueeze(1).expand(n_imgs, depth_pred.shape[1] * depth_pred.shape[2]).reshape(-1)

            n_pts = pts_img.shape[2]

            pts_hyp = torch.empty((n_imgs, 3, n * 2 + 1, n_pts), dtype=torch.float32, device=depth_pred.device)
            for i in range(-n, n + 1):
                pts_h = pts_img * (depth_pred.view(n_imgs, 1, -1) + i * offset)
                pts_h = torch.bmm(R_T, torch.bmm(K_inv, pts_h) - tvecs[ref_idx].unsqueeze(-1))
                pts_hyp[..., i + n, :] = pts_h

            # SECOND we re-project those points into the corresponding src images to calculate the variance feature
            n_hpts = (n*2+1) * n_pts
            w = torch.ones((n_imgs, 1, n_hpts), dtype=torch.float32).type_as(pts_hyp)
            pts_H = torch.cat((pts_hyp.view(n_imgs, 3, n_hpts), w), dim=1)

            # building projection matrix
            P = torch.cat((rotmats, tvecs[..., None]), dim=2)
            P = torch.bmm(K, P)

            pts_img_H = torch.bmm(P[ref_src_edges[1]], pts_H[gather_idx])
            z_buffer = pts_img_H[:, 2]
            z_buffer = torch.abs(z_buffer) + 1e-8  # add safety to ensure no div/0
            pts_img = pts_img_H[:, :2] / z_buffer[:, None]

            grid = pts_img.transpose(2, 1).view(pts_img.shape[0], n_hpts, 1, 2)
            grid[..., 0] = (grid[..., 0] / float(self.hparams.img_size[1] - 1)) * 2 - 1.0  # normalize to [-1, 1]
            grid[..., 1] = (grid[..., 1] / float(self.hparams.img_size[0] - 1)) * 2 - 1.0  # normalize to [-1, 1]

        x = F.grid_sample(img_feats[ref_src_edges[1]], grid, mode='bilinear', align_corners=True)
        x = x.squeeze(3)
        x_avg = scatter(x, gather_idx, dim=0, reduce='mean')
        x_avg_sq = scatter(x ** 2, gather_idx, dim=0, reduce='mean')
        x_var = x_avg_sq - x_avg ** 2

        pts_feat = x_var\
            .view(n_imgs, self.hparams.feat_dim, 2*n+1, n_pts)\
            .permute(0, 3, 2, 1)\
            .reshape(n_pts*n_imgs, 2*n+1, self.hparams.feat_dim)
        pts_hyp = pts_hyp.permute(0, 3, 2, 1).reshape(n_pts*n_imgs, 2*n+1, 3)

        offset_preds = self.decoder(xs, pts_hyp, pts_feat, pts_batch)
        offset_vals = torch.linspace(-n*offset, n*offset, 2*n+1)\
            .type_as(offset_preds).unsqueeze(0)\
            .expand(n_pts*n_imgs, 2*n+1)
        offset = torch.sum(offset_vals*offset_preds, dim=1).view(n_imgs, *depth_pred.shape[1:])
        return offset

    def log_metrics(self, metrics, prefix='val'):
        self.log(prefix + '/loss_2d', metrics['loss_2d'], on_epoch=True, sync_dist=True)
        self.log(prefix + '/loss', metrics['loss'], on_epoch=True, sync_dist=True)

        for k, v in metrics['initial'].items():
            self.log(prefix + '_2d/' + k, v, on_epoch=True, sync_dist=True)

        if 'final' in metrics.keys():
            for k, v in metrics['final'].items():
                self.log(prefix + '_final/' + k, v, on_epoch=True, sync_dist=True)

        if 'half' in metrics.keys():
            for k, v in metrics['half'].items():
                self.log(prefix + '_half/' + k, v, on_epoch=True, sync_dist=True)

        if 'quarter' in metrics.keys():
            for k, v in metrics['quarter'].items():
                self.log(prefix + '_quarter/' + k, v, on_epoch=True, sync_dist=True)

        for i, metrics_i in enumerate(metrics['ref']):
            prefix_i = prefix + '_ref{}'.format(i)
            for k, v in metrics_i.items():
                self.log(prefix_i + '/' + k, v, on_epoch=True, sync_dist=True)

        return

    def training_step(self, batch, batch_idx):
        if not self.hparams.finetune:
            self.mvsnet.feat_extractor.eval()
        offsets = [0.05, 0.05, 0.025]
        n_iters = 1 if self.current_epoch < 20 and not self.hparams.finetune else 2
        out = self.forward(batch, offsets, n_iters)
        self.log_metrics(out, prefix='train')
        return out['loss']

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch, [0.05, 0.05, 0.025], 2)
        self.log_metrics(out, prefix='val')
        return out['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.mvsnet.parameters()},
            {'params': self.pointnet.parameters()},
            {'params': self.sparse_conv.parameters()},
            {'params': self.decoder.parameters()},
            {'params': self.refine_quarter.parameters()},
            {'params': self.refine_half.parameters()},
            {'params': self.refine_full.parameters()},
        ], lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step,
                                                    gamma=self.hparams.lr_gamma)
        return [optimizer], [scheduler]
