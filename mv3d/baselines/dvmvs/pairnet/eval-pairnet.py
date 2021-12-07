from mv3d.baselines.dvmvs.pairnet import model
from mv3d.baselines.dvmvs.config import Config
from mv3d.baselines.dvmvs.utils import get_warp_grid_for_cost_volume_calculation, cost_volume_fusion
import torch
import numpy as np
import os
from mv3d.eval.main import main

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

warp_grid = get_warp_grid_for_cost_volume_calculation(width=int(Config.test_image_width / 2),
                                                      height=int(Config.test_image_height / 2),
                                                      device=DEVICE)

min_depth = 0.25
max_depth = 20.0
n_depth_levels = 64


class Pairnet(torch.nn.Module):
    def __init__(self):
        super(Pairnet, self).__init__()
        self.feature_extractor = model.FeatureExtractor()
        self.feature_shrinker = model.FeatureShrinker()
        self.cost_volume_encoder = model.CostVolumeEncoder()
        self.cost_volume_decoder = model.CostVolumeDecoder()


def load_pairnet(weights_folder=os.path.join(os.path.dirname(__file__), 'pairnet_weights')):
    pairnet = Pairnet()
    pairnet.feature_extractor.load_state_dict(torch.load(os.path.join(weights_folder, '0_feature_extractor')))
    pairnet.feature_shrinker.load_state_dict(torch.load(os.path.join(weights_folder, '1_feature_pyramid')))
    pairnet.cost_volume_encoder.load_state_dict(torch.load(os.path.join(weights_folder, '2_encoder')))
    pairnet.cost_volume_decoder.load_state_dict(torch.load(os.path.join(weights_folder, '3_decoder')))
    return pairnet


def process_scene(batch, scene, dset, net):
    with torch.no_grad():

        ref_idx = torch.unique(batch.ref_src_edges[0])
        half_K = batch.K.clone()
        half_K[:, 0:2, :] = half_K[:, 0:2, :] / 2.
        poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2)
        eye_row = torch.tensor([[[0., 0., 0., 1.]]]).type_as(poses).repeat(poses.shape[0], 1, 1)
        poses = torch.cat((poses, eye_row), dim=1)
        poses = poses.inverse()
        depth_preds = []

        for r_idx in ref_idx:
            src_idx = batch.ref_src_edges[1, batch.ref_src_edges[0] == r_idx]
            src_idx = src_idx[src_idx != r_idx]

            src_feature_halfs = []
            for s_idx in src_idx:
                src_feature_half, _, _, _ = net.feature_shrinker(*net.feature_extractor(batch.images[s_idx, None]))
                src_feature_halfs.append(src_feature_half)

            reference_feature_half, reference_feature_quarter, \
            reference_feature_one_eight, reference_feature_one_sixteen = net.feature_shrinker(
                *net.feature_extractor(batch.images[r_idx, None]))

            cost_volume = cost_volume_fusion(image1=reference_feature_half,
                                             image2s=src_feature_halfs,
                                             pose1=poses[r_idx].unsqueeze(0),
                                             pose2s=poses[src_idx].unsqueeze(1).unbind(dim=0),
                                             K=half_K[r_idx].unsqueeze(0),
                                             warp_grid=warp_grid,
                                             min_depth=min_depth,
                                             max_depth=max_depth,
                                             n_depth_levels=n_depth_levels,
                                             device=DEVICE,
                                             dot_product=True)

            skip0, skip1, skip2, skip3, bottom = net.cost_volume_encoder(
                features_half=reference_feature_half,
                features_quarter=reference_feature_quarter,
                features_one_eight=reference_feature_one_eight,
                features_one_sixteen=reference_feature_one_sixteen,
                cost_volume=cost_volume)

            depth_pred, _, _, _, _ = net.cost_volume_decoder(batch.images[r_idx, None], skip0, skip1, skip2, skip3,
                                                             bottom)
            depth_pred = depth_pred.cpu().numpy()
            depth_preds.append(depth_pred)

        depth_preds = np.concatenate(depth_preds, axis=0)

        return depth_preds, None, None


if __name__ == '__main__':
    print('Loading model...')
    net = load_pairnet()
    dset_kwargs = {
        'mean_rgb': [0.485, 0.456, 0.406],
        'std_rgb': [0.229, 0.224, 0.225],
        'scale_rgb': 255.,
        'img_size': (256, 320)
    }
    main('dvmvs_pairnet', process_scene, dset_kwargs, net, depth=True)
