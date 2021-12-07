from mv3d.baselines.gpmvs.lightningmodel import GPMVS
import torch
from mv3d.eval.main import main


def process_scene(batch, scene, dset, net):
    with torch.no_grad():
        _, depth_preds = net(batch)
        depth_preds = depth_preds.detach().cpu().numpy()

    return depth_preds, None, None


if __name__ == '__main__':
    print('Loading finetuned model...')
    net = GPMVS.load_from_checkpoint('gpmvs_weights/finetuned.ckpt')
    dset_kwargs = {
        'mean_rgb': [81., 81., 81.],
        'std_rgb': [35., 35., 35.],
        'scale_rgb': 1.,
        'img_size': (256, 320)
    }
    main('gpmvs_ft', process_scene, dset_kwargs, net, depth=True)

    print('Loading pretrained model...')
    net = GPMVS()
    net.load_pretrained_gpmvs()
    main('gpmvs', process_scene, dset_kwargs, net, depth=True)
