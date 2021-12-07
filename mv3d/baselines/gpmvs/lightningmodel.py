import pytorch_lightning as pl
from mv3d.baselines.gpmvs.GPlayer import GPlayer
from mv3d.baselines.gpmvs.enCoder import enCoder
from mv3d.baselines.gpmvs.deCoder import deCoder
from mv3d.baselines.gpmvs.loss_functions import compute_errors
from mv3d.baselines.gpmvs import utils
import torch
import os
import numpy as np
from mv3d.eval.metricfunctions import calc_2d_depth_metrics


class GPMVS(pl.LightningModule):
    def __init__(self, lr=5e-5, lr_step=100, lr_gamma=0.1, momentum=0.9, beta=0.999, weight_decay=0.):
        super().__init__()
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.momentum = momentum
        self.beta = beta
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.encoder = enCoder()
        self.decoder = deCoder()
        self.gplayer = GPlayer()

        pixel_coordinate = np.indices([320, 256]).astype(np.float32)
        pixel_coordinate = np.concatenate(
            (pixel_coordinate, np.ones([1, 320, 256])), axis=0)
        pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])
        self.pixel_coordinate = torch.from_numpy(pixel_coordinate)

    def load_pretrained_gpmvs(self, weights_folder=os.path.join(os.path.dirname(__file__), 'gpmvs_weights', 'pretrained')):
        enc_weights = torch.load(os.path.join(weights_folder, 'encoder_model_best.pth'))
        enc_state_dict = {k[7:]: enc_weights['state_dict'][k] for k in enc_weights['state_dict'].keys()}
        self.encoder.load_state_dict(enc_state_dict)

        dec_weights = torch.load(os.path.join(weights_folder, 'decoder_model_best.pth'))
        dec_state_dict = {k[7:]: dec_weights['state_dict'][k] for k in dec_weights['state_dict'].keys()}
        self.decoder.load_state_dict(dec_state_dict)

        gp_weights = torch.load(os.path.join(weights_folder, 'gp_model_best.pth'))
        self.gplayer.load_state_dict(gp_weights['state_dict'])
        print('Loaded pretrained weights from file')

    def forward(self, batch):
        # get values to use ref_src_edges data structure in for-loop style forward pass
        ref_idx = torch.unique(batch.ref_src_edges[0])
        n_batches = batch.images_batch.max().cpu().item() + 1

        inv_depth_gt = torch.where(batch.depth_images != 0., 1. / batch.depth_images,
                                   torch.tensor(0.).type_as(batch.depth_images))

        poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2)
        eye_row = torch.tensor([0., 0., 0., 1.]).type_as(poses).unsqueeze(0).repeat(poses.shape[0], 1).unsqueeze(1)
        poses = torch.cat((poses, eye_row), dim=1)

        ref_poses = []
        idepths = []
        latents = []
        conv1s = []
        conv2s = []
        conv3s = []
        conv4s = []

        # generate initial latent encodings
        for r_idx in ref_idx:
            src_idx = batch.ref_src_edges[1, batch.ref_src_edges[0] == r_idx]
            src_idx = src_idx[src_idx != r_idx]
            psv = 0.
            for s_idx in src_idx:
                psv += self.get_psv(batch.images[r_idx], batch.images[s_idx], poses[r_idx], poses[s_idx], batch.K[0])
            psv /= src_idx.shape[0]
            conv5, conv4, conv3, conv2, conv1 = self.encoder.getConvs(batch.images[r_idx].unsqueeze(0), psv)

            if not self.training:  # save space on GPU when num of input images is large (>1000)
                conv5 = conv5.detach().cpu()
                conv4 = conv4.detach().cpu()
                conv3 = conv3.detach().cpu()
                conv2 = conv2.detach().cpu()
                conv1 = conv1.detach().cpu()

            ref_poses.append(poses[r_idx].cpu().numpy())
            conv1s.append(conv1)
            conv2s.append(conv2)
            conv3s.append(conv3)
            conv4s.append(conv4)
            latents.append(conv5)
        ref_poses = np.stack(ref_poses, axis=0)

        # adjust latent layer
        Y = torch.cat(latents, dim=0).unsqueeze(0).to(self.device)
        Z = []
        for b in range(n_batches):
            batch_idx = batch.images_batch[ref_idx] == b
            D = torch.from_numpy(utils.genDistM(ref_poses[batch_idx.cpu().numpy()])).type_as(Y).unsqueeze(0)
            Z.append(self.gplayer(D, Y[:1, batch_idx]).squeeze(0))
        Z = torch.cat(Z, dim=0)
        b, l, c, h, w = Y.size()

        # generate final encodings
        loss = 0.
        for i in range(ref_idx.shape[0]):
            conv5 = Z[i].view(1, c, h, w)
            conv4 = conv4s[i].to(self.device)
            conv3 = conv3s[i].to(self.device)
            conv2 = conv2s[i].to(self.device)
            conv1 = conv1s[i].to(self.device)
            pred = self.decoder(conv5, conv4, conv3, conv2, conv1)
            loss += compute_errors(inv_depth_gt[i].unsqueeze(0), pred)
            idepths.append(pred[0][0, 0])
        loss /= len(idepths)

        idepths = torch.stack(idepths, dim=0)
        idepths = torch.clamp(idepths, 0.02, 2.)   # clamp to range (0.5m, 50m)
        depth_preds = 1. / idepths

        return loss, depth_preds

    def get_psv(self, r_img, n_img, r_pose, n_pose, K):

        left_image = r_img
        right_image = n_img

        left_pose = r_pose
        right_pose = n_pose

        camera_k = K

        left2right = right_pose @ left_pose.inverse()

        left_in_right_T = left2right[0:3, 3]
        left_in_right_R = left2right[0:3, 0:3]
        K = camera_k
        K_inverse = K.inverse()
        KRK_i = K @ (left_in_right_R @ K_inverse)
        KRKiUV = KRK_i @ (self.pixel_coordinate.type_as(r_img))
        KT = K @ left_in_right_T

        left_image = left_image.unsqueeze(0)
        right_image = right_image.unsqueeze(0)
        KRKiUV = KRKiUV.unsqueeze(0)
        KT = KT.unsqueeze(0).unsqueeze(2)

        return self.encoder.getVolume(left_image, right_image, KRKiUV, KT)

    def log_metrics(self, metrics, prefix='val'):
        for k, v in metrics.items():
            self.log(prefix + '_final/' + k, v, on_epoch=True, sync_dist=True)
        return

    def training_step(self, batch, batch_idx):
        loss, depth_preds = self.forward(batch)
        metrics = calc_2d_depth_metrics(depth_preds, batch.depth_images)
        metrics['loss_2d'] = loss
        self.log_metrics(metrics, prefix='train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, depth_preds = self.forward(batch)
        metrics = calc_2d_depth_metrics(depth_preds, batch.depth_images)
        metrics['loss_2d'] = loss
        self.log_metrics(metrics, prefix='val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     betas=(self.momentum, self.beta),
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gamma)
        return [optimizer], [scheduler]
