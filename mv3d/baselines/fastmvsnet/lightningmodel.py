import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import os
from mv3d.eval.metricfunctions import calc_2d_depth_metrics
from mv3d.baselines.fastmvsnet.model import FastMVSNet, PointMVSNetLoss


class FastMVSNetPLModel(pl.LightningModule):
    def __init__(self, lr=5e-5, lr_step=100, lr_gamma=0.1):
        super().__init__()
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.save_hyperparameters()

        self.fastmvsnet = FastMVSNet()
        self.pmvsnet_loss = PointMVSNetLoss(8.)  # valid threshold appears to be set upper bound on abs diff?
        self.pts_mean = torch.tensor([1.97145182, -1.52387525, 651.07223895])
        self.pts_std = torch.tensor([84.45612252, 93.22252387, 80.08551226])

    def load_pretrained_fastmvsnet(self, weights_folder=os.path.join(os.path.dirname(__file__), 'fmvs_weights')):
        ckpt = torch.load(os.path.join(weights_folder, 'pretrained.pth'))
        # pretrained weights store upsampling network as 'upnet' instead of 'propagation_net'
        state_dict = {k[7:].replace('upnet', 'propagation_net'): ckpt['model'][k] for k in ckpt['model'].keys()}
        self.fastmvsnet.load_state_dict(state_dict)
        print('Loaded pretrained weights from file')

    def forward(self, batch, img_scales, inter_scales, depth_start, depth_interval, n_depth, move_to_cpu=False,
                use_gt_depth=False):

        # get values to use ref_src_edges data structure in for-loop style forward pass
        ref_idx = torch.unique(batch.ref_src_edges[0])
        n_ref_imgs = ref_idx.shape[0]

        preds_list = []
        depth_intervals = torch.ones(n_ref_imgs).type_as(batch.depth_images) * depth_interval
        # depth_intervals = torch.zeros(n_ref_imgs).type_as(batch.depth_images)
        for i in range(n_ref_imgs):
            r_idx = ref_idx[i]
            s_idx = batch.ref_src_edges[1, batch.ref_src_edges[0] == r_idx]
            s_idx = s_idx[s_idx != r_idx]  # remove ref index from src idx list
            n_views = 1 + s_idx.shape[0]

            img_list = torch.cat((batch.images[r_idx].unsqueeze(0), batch.images[s_idx]), dim=0).unsqueeze(0)
            rotmats = torch.cat((batch.rotmats[r_idx].unsqueeze(0), batch.rotmats[s_idx]), dim=0).unsqueeze(0)
            tvecs = torch.cat((batch.tvecs[r_idx].unsqueeze(0), batch.tvecs[s_idx]), dim=0).unsqueeze(0)
            K = torch.cat((batch.K[r_idx].unsqueeze(0), batch.K[s_idx]), dim=0).unsqueeze(0)

            if use_gt_depth:
                # use GT min and max depths to determine depth start/interval
                with torch.no_grad():
                    depth_gt = batch.depth_images[i]
                    valid = ~torch.eq(depth_gt, 0.0)
                    depth_start = depth_gt[valid].min()
                    depth_end = depth_gt[valid].max()
                    depth_interval = (depth_end - depth_start) / (n_depth - 1)
                    depth_intervals[i] = depth_interval

            cam_params_list = torch.zeros((1, n_views, 2, 4, 4)).type_as(rotmats)
            cam_params_list[0, :, 0, :3, :3] = rotmats
            cam_params_list[0, :, 0, :3, 3] = tvecs
            cam_params_list[0, :, 1, :3, :3] = K
            cam_params_list[0, 0, 1, 3, 0] = depth_start
            cam_params_list[0, 0, 1, 3, 1] = depth_interval
            cam_params_list[0, 0, 1, 3, 2] = n_depth

            data_batch = {
                "img_list": img_list,
                "cam_params_list": cam_params_list,
                'mean': self.pts_mean.to(self.device),
                'std': self.pts_std.to(self.device)
            }
            preds = self.fastmvsnet(data_batch, img_scales, inter_scales, isGN=True, isTest=not self.training)
            if move_to_cpu:
                for k in preds.keys():
                    preds[k] = preds[k].detach().cpu()
            preds_list.append(preds)

        out = dict()
        for k in preds_list[0].keys():
            v = torch.cat([p[k] for p in preds_list], dim=0)
            out[k] = v

        return out, depth_intervals

    def log_metrics(self, metrics, prefix='val'):
        self.log(prefix + '/loss_2d', metrics['loss_2d'], on_epoch=True, sync_dist=True)
        self.log(prefix + '/loss', metrics['loss'], on_epoch=True, sync_dist=True)

        for k, v in metrics['initial'].items():
            self.log(prefix + '_2d/' + k, v, on_epoch=True, sync_dist=True)

        if 'final' in metrics.keys():
            for k, v in metrics['final'].items():
                self.log(prefix + '_final/' + k, v, on_epoch=True, sync_dist=True)

        for i, metrics_i in enumerate(metrics['ref']):
            prefix_i = prefix + '_ref{}'.format(i)
            for k, v in metrics_i.items():
                self.log(prefix_i + '/' + k, v, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        n_depth = 48

        out, depth_intervals = self.forward(batch, (0.125, 0.25), (0.75, 0.375), 0.5, 0.1, n_depth)
        loss_dict = self.pmvsnet_loss(out, batch.depth_images.unsqueeze(1), depth_intervals, isFlow=True)
        losses = sum(loss_dict.values())

        log_dict = {
            'loss_2d': loss_dict['coarse_loss'],
            'loss': losses,
            'ref': []
        }

        coarse_pred = out['coarse_depth_map'].squeeze(1)
        flow1 = out['flow1'].squeeze(1)
        flow2 = out['flow2'].squeeze(1)
        coarse_gt = F.interpolate(batch.depth_images.unsqueeze(1), coarse_pred.shape[-2:], mode='nearest').squeeze(1)
        flow1_gt = F.interpolate(batch.depth_images.unsqueeze(1), flow1.shape[-2:], mode='nearest').squeeze(1)
        flow2_gt = F.interpolate(batch.depth_images.unsqueeze(1), flow2.shape[-2:], mode='nearest').squeeze(1)

        coarse_metrics = calc_2d_depth_metrics(coarse_pred, coarse_gt)
        log_dict['initial'] = coarse_metrics

        flow1_metrics = calc_2d_depth_metrics(flow1, flow1_gt)
        flow1_metrics['loss_2d'] = loss_dict['flow1_loss']
        log_dict['ref'].append(flow1_metrics)

        flow2_metrics = calc_2d_depth_metrics(flow2, flow2_gt)
        flow2_metrics['loss_2d'] = loss_dict['flow2_loss']
        log_dict['final'] = flow2_metrics

        self.log_metrics(log_dict, prefix='train')
        return losses

    def validation_step(self, batch, batch_idx):
        n_depth = 96
        out, depth_intervals = self.forward(batch, (0.125, 0.25), (0.75, 0.375), 0.5, 0.05, n_depth)
        loss_dict = self.pmvsnet_loss(out, batch.depth_images.unsqueeze(1), depth_intervals, isFlow=True)
        losses = sum(loss_dict.values())

        log_dict = {
            'loss_2d': loss_dict['coarse_loss'],
            'loss': losses,
            'ref': []
        }

        coarse_pred = out['coarse_depth_map'].squeeze(1)
        flow1 = out['flow1'].squeeze(1)
        flow2 = out['flow2'].squeeze(1)
        coarse_gt = F.interpolate(batch.depth_images.unsqueeze(1), coarse_pred.shape[-2:], mode='nearest').squeeze(1)
        flow1_gt = F.interpolate(batch.depth_images.unsqueeze(1), flow1.shape[-2:], mode='nearest').squeeze(1)
        flow2_gt = F.interpolate(batch.depth_images.unsqueeze(1), flow2.shape[-2:], mode='nearest').squeeze(1)

        coarse_metrics = calc_2d_depth_metrics(coarse_pred, coarse_gt)
        log_dict['initial'] = coarse_metrics

        flow1_metrics = calc_2d_depth_metrics(flow1, flow1_gt)
        flow1_metrics['loss_2d'] = loss_dict['flow1_loss']
        log_dict['ref'].append(flow1_metrics)

        flow2_metrics = calc_2d_depth_metrics(flow2, flow2_gt)
        flow2_metrics['loss_2d'] = loss_dict['flow2_loss']
        log_dict['final'] = flow2_metrics

        self.log_metrics(log_dict, prefix='val')
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gamma)
        return [optimizer], [scheduler]
