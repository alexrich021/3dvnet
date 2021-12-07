from mv3d.baselines.dvmvs.fusionnet import model
from mv3d.baselines.dvmvs.config import Config
from mv3d.baselines.dvmvs.utils import get_warp_grid_for_cost_volume_calculation, cost_volume_fusion, \
    get_non_differentiable_rectangle_depth_estimation
from mv3d.eval.main import main
import torch
import numpy as np
import os

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

warp_grid = get_warp_grid_for_cost_volume_calculation(width=int(Config.test_image_width / 2),
                                                      height=int(Config.test_image_height / 2),
                                                      device=DEVICE)

min_depth = 0.25
max_depth = 20.0
n_depth_levels = 64


class Fusionnet(torch.nn.Module):
    def __init__(self):
        super(Fusionnet, self).__init__()
        self.feature_extractor = model.FeatureExtractor()
        self.feature_shrinker = model.FeatureShrinker()
        self.cost_volume_encoder = model.CostVolumeEncoder()
        self.lstm_fusion = model.LSTMFusion()
        self.cost_volume_decoder = model.CostVolumeDecoder()


def load_fusionnet(weights_folder=os.path.join(os.path.dirname(__file__), 'fusionnet_weights')):
    fusionnet = Fusionnet()
    fusionnet.feature_extractor.load_state_dict(torch.load(os.path.join(weights_folder, '0_feature_extractor')))
    fusionnet.feature_shrinker.load_state_dict(torch.load(os.path.join(weights_folder, '1_feature_pyramid')))
    fusionnet.cost_volume_encoder.load_state_dict(torch.load(os.path.join(weights_folder, '2_encoder')))
    fusionnet.lstm_fusion.load_state_dict(torch.load(os.path.join(weights_folder, '3_lstm_fusion')))
    fusionnet.cost_volume_decoder.load_state_dict(torch.load(os.path.join(weights_folder, '4_decoder')))
    return fusionnet


def process_scene(batch, scene, dset, net):
    with torch.no_grad():

        ref_idx = torch.unique(batch.ref_src_edges[0])
        half_K = batch.K.clone()
        half_K[:, 0:2, :] = half_K[:, 0:2, :] / 2.
        lstm_K_bottom = batch.K.clone().cuda()
        lstm_K_bottom[:, 0:2, :] = lstm_K_bottom[:, 0:2, :] / 32.0
        poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2)
        eye_row = torch.tensor([[[0., 0., 0., 1.]]]).type_as(poses).repeat(poses.shape[0], 1, 1)
        poses = torch.cat((poses, eye_row), dim=1)
        poses = poses.inverse()
        depth_preds = []

        lstm_state = None
        previous_depth = None
        previous_pose = None
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
            ref_pose = poses[r_idx].unsqueeze(0)

            cost_volume = cost_volume_fusion(image1=reference_feature_half,
                                             image2s=src_feature_halfs,
                                             pose1=ref_pose,
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

            if previous_depth is not None:
                depth_estimation = get_non_differentiable_rectangle_depth_estimation(
                    reference_pose_torch=ref_pose,
                    measurement_pose_torch=previous_pose,
                    previous_depth_torch=previous_depth,
                    full_K_torch=batch.K[r_idx].unsqueeze(0),
                    half_K_torch=half_K[r_idx].unsqueeze(0),
                    original_height=Config.test_image_height,
                    original_width=Config.test_image_width)
                depth_estimation = torch.nn.functional.interpolate(input=depth_estimation,
                                                                   scale_factor=(1.0 / 16.0),
                                                                   mode="nearest")
            else:
                depth_estimation = torch.zeros(
                    size=(1, 1, int(Config.test_image_height / 32.0), int(Config.test_image_width / 32.0))).to(DEVICE)

            lstm_state = net.lstm_fusion(current_encoding=bottom,
                                         current_state=lstm_state,
                                         previous_pose=previous_pose,
                                         current_pose=ref_pose,
                                         estimated_current_depth=depth_estimation,
                                         camera_matrix=lstm_K_bottom[r_idx].unsqueeze(0))

            depth_pred, _, _, _, _ = net.cost_volume_decoder(batch.images[r_idx, None], skip0, skip1, skip2, skip3,
                                                             lstm_state[0])
            previous_depth = depth_pred.view(1, 1, Config.test_image_height, Config.test_image_width)
            previous_pose = ref_pose

            depth_pred = depth_pred.cpu().numpy()
            depth_preds.append(depth_pred)

        depth_preds = np.concatenate(depth_preds, axis=0)

        return depth_preds, None, None


if __name__ == '__main__':
    print('Loading model...')
    net = load_fusionnet()
    dset_kwargs = {
        'mean_rgb': [0.485, 0.456, 0.406],
        'std_rgb': [0.229, 0.224, 0.225],
        'scale_rgb': 255.,
        'img_size': (256, 320)
    }
    main('dvmvs_fusionnet', process_scene, dset_kwargs, net, depth=True)
