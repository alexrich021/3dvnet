from mv3d.baselines.fastmvsnet.lightningmodel import FastMVSNetPLModel
from mv3d.eval.main import main
import torch


def process_scene(batch, scene, dset, net):
    with torch.no_grad():

        out, _ = net(batch, (0.125, 0.25, 0.5), (1.0, 0.75, 0.15), 0.5, 0.05, 96, move_to_cpu=True)
        depth_preds = out['flow3']
        prob_map = out['coarse_prob_map']
        depth_preds = depth_preds.squeeze(1).detach().cpu().numpy()
        prob_map = prob_map.squeeze(1).detach().cpu().numpy()

        return depth_preds, prob_map, None


if __name__ == '__main__':
    print('Loading finetuned model...')
    net = FastMVSNetPLModel.load_from_checkpoint('fmvs_weights/finetuned.ckpt')
    dset_kwargs = {
        'mean_rgb': [0., 0., 0.],
        'std_rgb': [1., 1., 1.],
        'scale_rgb': 1.,
        'img_size': (448, 640)
    }
    main('fmvs_ft', process_scene, dset_kwargs, net, depth=True)

    print('Loading pretrained model...')
    net = FastMVSNetPLModel()
    net.load_pretrained_fastmvsnet()
    main('fmvs', process_scene, dset_kwargs, net, depth=True)
