from mv3d.baselines.pointmvsnet.lightningmodel import PointMVSNetPLModel
from mv3d.eval.main import main
from mv3d.utils import get_propability_map_from_flow
import torch


def process_scene(batch, scene, dset, net):
    with torch.no_grad():

        out, _ = net(batch, (0.125, 0.25, 0.5), (1.0, 0.75, 0.15), 0.5, 0.05, 96, move_to_cpu=True)
        depth_preds = out['flow3']
        flow_prob = get_propability_map_from_flow(out['flow3_prob'])
        init_prob_map = out["coarse_prob_map"]
        depth_preds = depth_preds.squeeze(1).detach().cpu().numpy()
        init_prob_map = init_prob_map.squeeze(1).detach().cpu().numpy()
        flow_prob = flow_prob.squeeze(1).detach().cpu().numpy()

        return depth_preds, init_prob_map, flow_prob


if __name__ == '__main__':
    print('Loading finetuned model...')
    net = PointMVSNetPLModel.load_from_checkpoint('pmvs_weights/finetuned.ckpt')
    dset_kwargs = {
        'mean_rgb': [0., 0., 0.],
        'std_rgb': [1., 1., 1.],
        'scale_rgb': 1.,
        'img_size': (448, 640)
    }
    main('pmvs_ft', process_scene, dset_kwargs, net, depth=True)

    print('Loading pretrained model...')
    net = PointMVSNetPLModel()
    net.load_pretrained_pointmvsnet()
    main('pmvs', process_scene, dset_kwargs, net, depth=True)
