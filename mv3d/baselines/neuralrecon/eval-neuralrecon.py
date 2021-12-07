from mv3d.baselines.neuralrecon.models import NeuralRecon
from mv3d.baselines.neuralrecon.config import cfg, update_config
from mv3d.baselines.neuralrecon.datasets import transforms
from mv3d.eval.main import main
import open3d as o3d
from skimage.measure import marching_cubes_lewiner
import torch
import numpy as np
import os
import json

neuralrecon_transforms = transforms.Compose([
    transforms.RandomTransformSpace([96, 96, 96], 0.04, False, False, 0, 0),
    transforms.IntrinsicsPoseToProjection(9, 4)
])


class Args:
    cfg = 'config/test.yaml'
    opts = []

update_config(cfg, Args())


def process_scene(batch, scene, dset, net):
    scene_name = os.path.basename(scene)
    scene_info = json.load(open(os.path.join(scene, 'info.json'), 'r'))
    gt_mesh = o3d.io.read_point_cloud(scene_info['gt_mesh'])
    min_bound = np.min(np.asarray(gt_mesh.points), axis=0)
    origin = min_bound - 0.25

    with torch.no_grad():
        n_imgs = batch.images.shape[0]
        batch = batch.to(torch.device('cpu'))  # neuralrecon transforms expect CPU tensors as input

        poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2)
        eye_row = torch.tensor([[[0., 0., 0., 1.]]]).repeat(n_imgs, 1, 1).type_as(poses)
        poses = torch.cat((poses, eye_row), dim=1)
        poses_inv = torch.inverse(poses)   # neuralrecon expects cam2world poses

        n_fragments = (n_imgs - 1) // 9 + 1
        for i in range(n_fragments):
            end_idx = (i + 1) * 9
            if end_idx >= n_imgs:   # make sure final fragment still contains 9 views
                end_idx = n_imgs - 1
                start_idx = end_idx - 9
            else:
                start_idx = i * 9
            save_scene = i == n_fragments - 1
            neuralrecon_inputs = {
                'imgs': batch.images[start_idx: end_idx, [2, 1, 0]],  # neuralrecon expects rgb images
                'intrinsics': batch.K[start_idx: end_idx],
                'extrinsics': poses_inv[start_idx: end_idx],
                'vol_origin': origin,
                'scene': scene_name,
                'fragment': scene_name + '_' + str(i),
                'epoch': 0
            }
            neuralrecon_inputs = neuralrecon_transforms(neuralrecon_inputs)

            # make input dict a valid batch of size 1
            for k, v in neuralrecon_inputs.items():
                if type(v) == torch.Tensor:
                    neuralrecon_inputs[k] = v.unsqueeze(0)
                else:
                    neuralrecon_inputs[k] = [v, ]

            outputs, _ = net(neuralrecon_inputs, save_scene)

            if save_scene:
                tsdf = outputs['scene_tsdf'][0].cpu().numpy()
                tsdf_origin = outputs['origin'][0].cpu().numpy()
                verts, faces, _, _ = marching_cubes_lewiner(tsdf, 0., (0.04, 0.04, 0.04), gradient_direction='descent')
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts + tsdf_origin)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_triangle_normals()
                mesh.compute_vertex_normals()

                return mesh


if __name__ == '__main__':
    print('Loading model...')
    state_dict = torch.load('neuralrecon_weights/model_000047.ckpt')
    model_state_dict = {k[7:]: state_dict['model'][k] for k in state_dict['model'].keys()}
    net = NeuralRecon(cfg)
    net.load_state_dict(model_state_dict)
    dset_kwargs = {
        'mean_rgb': [0., 0., 0.],
        'std_rgb': [1., 1., 1.],
        'scale_rgb': 1.,
        'img_size': (480, 640)
    }
    main('neuralrecon', process_scene, dset_kwargs, net, depth=False)
