from mv3d.baselines.atlas.model import VoxelNet
import open3d as o3d
import json
from skimage.measure import marching_cubes_lewiner
import torch
import numpy as np
import os
from mv3d.eval.main import main

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def process_scene(batch, scene, dset, net):
    scene_info = json.load(open(os.path.join(scene, 'info.json'), 'r'))
    gt_mesh = o3d.io.read_triangle_mesh(scene_info['gt_mesh'])
    min_bound = np.min(np.asarray(gt_mesh.vertices), axis=0)
    origin = min_bound - 0.25
    net.origin = torch.tensor(origin).unsqueeze(0).float()

    with torch.no_grad():

        poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2)
        projection = torch.bmm(batch.K, poses)

        atlas_batch = {
            'image': batch.images.unsqueeze(0),
            'projection': projection.unsqueeze(0)
        }
        out, _ = net(atlas_batch)
        tsdf = out['vol_04_tsdf'][0, 0].cpu().numpy()

        verts, faces, _, _ = marching_cubes_lewiner(tsdf, 0., (0.04, 0.04, 0.04), gradient_direction='ascent')
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts + origin)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        return mesh


if __name__ == '__main__':
    print('Loading model...')
    net = VoxelNet.load_from_checkpoint('atlas_weights/final.ckpt')
    dset_kwargs = {
        'mean_rgb': [0., 0., 0.],
        'std_rgb': [1., 1., 1.],
        'scale_rgb': 1.,
        'img_size': (480, 640)
    }
    main('atlas', process_scene, dset_kwargs, net, depth=False)
