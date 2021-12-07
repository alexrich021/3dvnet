from mv3d.baselines.neuralrecon.models import NeuralRecon
from mv3d.baselines.neuralrecon.config import cfg, update_config
from mv3d.baselines.neuralrecon.datasets import transforms
from mv3d.baselines.neuralrecon.tools.tsdf_fusion.fusion import get_view_frustum
from mv3d.dsets import dataset
from mv3d.dsets import scenelists
from mv3d.eval import processresults
from mv3d.eval import meshtodepth
from mv3d.dsets import frameselector
import open3d as o3d
from skimage.measure import marching_cubes_lewiner
import torch
import numpy as np
import os
import argparse
from mv3d import config
from mv3d.baselines.neuralrecon.datasets import scannet

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def process_scene(scene_idx, scene, scene_save_dir, dset, net, overwrite=False):

    scene_name = os.path.basename(scene)
    gt_mesh = o3d.io.read_triangle_mesh(os.path.join(scene, '{}_vh_clean_2.ply'.format(scene_name)))
    min_bound = np.min(np.asarray(gt_mesh.vertices), axis=0)
    # origin = min_bound - 0.25

    nc_transforms = transforms.Compose([
        transforms.ResizeImage((640, 480)),
        transforms.ToTensor(),
        transforms.RandomTransformSpace([96, 96, 96], 0.04, False, False, 0, 0, max_epoch=991),
        transforms.IntrinsicsPoseToProjection(9, 4)
    ])
    nc_dset = scannet.ScanNetDataset('/home/alex/Desktop/scannet_copy', 'train', nc_transforms, 9, 2)

    with torch.no_grad():
        # batch, _, _, img_idx = dset.get(scene_idx, return_verbose=True, seed_idx=0)
        # batch.__setattr__('images_batch', torch.zeros(batch.images.shape[0], dtype=torch.long))
        #
        # n_imgs = batch.images.shape[0]
        #
        # poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2)
        # eye_row = torch.tensor([[[0., 0., 0., 1.]]]).repeat(n_imgs, 1, 1)
        # poses = torch.cat((poses, eye_row), dim=1)
        # poses = torch.inverse(poses)   # neuralrecon expects cam2world poses
        #
        # fragments = get_fragments(batch.K[0].numpy(), batch.depth_images.numpy(), poses.numpy(), origin,
        #                           cfg.TEST.N_VIEWS)

        for i, neuralrecon_inputs in enumerate(nc_dset):
            neuralrecon_inputs.pop('occ_list')
            neuralrecon_inputs.pop('tsdf_list')
            save_scene = i == len(nc_dset) - 1
            # print(neuralrecon_inputs['vol_origin_partial'])
            # neuralrecon_inputs = {
            #     'imgs': batch.images[fragment['image_ids']][:, [2, 1, 0]],  # neuralrecon expects rgb images
            #     'intrinsics': batch.K[fragment['image_ids']],
            #     'extrinsics': poses[fragment['image_ids']],
            #     'vol_origin': origin,
            #     'scene': scene_name,
            #     'fragment': scene_name + '_' + str(i),
            #     'epoch': 0
            # }
            # neuralrecon_inputs = nc_transforms(neuralrecon_inputs)
            # print(neuralrecon_inputs['vol_origin_partial'])

            # make input dict a valid batch of size 1
            for k, v in neuralrecon_inputs.items():
                if type(v) == torch.Tensor:
                    neuralrecon_inputs[k] = v.unsqueeze(0)
                else:
                    neuralrecon_inputs[k] = [v, ]

            # for k, v in neuralrecon_inputs.items():
            #     print(k, type(v))
            #     if type(v) == torch.Tensor:
            #         print('     ', v.shape)
            #     elif type(v) == list:
            #         print('     ', len(v))
            # exit()

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

                import vis_utils
                max_bnd = tsdf_origin + np.array([0.04*tsdf.shape[0], 0.04*tsdf.shape[1], 0.04*tsdf.shape[2]])
                # max_bnd = origin + np.array([0.5, 0.5, 0.5])
                cube = vis_utils.create_wireframe_rectangle(tsdf_origin, max_bnd)
                sphere = vis_utils.create_sphere(min_bound, 0.125)
                callbacks = dict()
                callbacks[ord('F')] = lambda x: vis_utils.remove_pcd(x, gt_mesh)
                callbacks[ord('C')] = lambda x: vis_utils.add_pcd(x, gt_mesh)
                o3d.visualization.draw_geometries_with_key_callbacks([mesh, cube, sphere], callbacks)
                exit()
                # o3d.io.write_triangle_mesh(mesh_file_path, mesh)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_interval', '-ii', type=int, default=20)
    parser.add_argument('--save_dir', '-d', type=str, default='/home/alex/Desktop/results/neuralrecon')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--cfg', type=str, default='/home/alex/Desktop/implicit_sfm/mv3d/baselines/neuralrecon/'
                                                   'config/test.yaml')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(cfg, args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('Preparing dataset...')
    scenes = scenelists.get_scenes_scannet(config.SCANNET_DIR, 'test')
    selector = frameselector.EveryNthSelector(5)
    # dset = dataset.Dataset(scenes, selector, None, (480, 640), (480, 640), False, n_src_on_either_side=1,
    #                        mean_rgb=[0., 0., 0.], std_rgb=[1., 1., 1.], scale_rgb=1)
    dset = dataset.Dataset(scenes, selector, None, (480, 640), (480, 640), False, n_src_on_either_side=1,
                           mean_rgb=[0., 0., 0.], std_rgb=[1., 1., 1.], scale_rgb=1)

    print('Loading model...')
    state_dict = torch.load('pretrained_weights/model_000047.ckpt')
    model_state_dict = {k[7:]: state_dict['model'][k] for k in state_dict['model'].keys()}
    net = NeuralRecon(cfg)
    net.load_state_dict(model_state_dict)
    net = net.to(DEVICE)
    net.eval()

    for j, scene in enumerate(scenes):
        scene_name = os.path.basename(scene)
        print('{} / {}: {}'.format(j + 1, len(scenes), scene_name))
        scene_save_dir = os.path.join(args.save_dir, 'scenes', scene_name)
        if not os.path.exists(scene_save_dir):
            os.makedirs(scene_save_dir)
        process_scene(j, scene, scene_save_dir, dset, net, args.overwrite)
        exit()
        processresults.process_scene_2d_metrics(scene, scene_save_dir, args.overwrite)
        processresults.process_volume_3d_metrics(scene, scene_save_dir, args.overwrite)
    processresults.calc_avg_metrics(args.save_dir)
