from mv3d.dsets import dataset, scenelists, frameselector
import torch
import os
import numpy as np
from mv3d import config
from mv3d.eval import processresults
from mv3d.eval import config as eval_config
from mv3d.eval import meshtodepth
import open3d as o3d


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(save_dirname, pred_func, dset_kwargs, net, overwrite=False, depth=True):

    save_dir = os.path.join(eval_config.SAVE_DIR, save_dirname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print('Preparing dataset...')
    if eval_config.DATASET_TYPE == 'scannet':
        scenes = sorted(scenelists.get_scenes_scannet(config.SCANNET_DIR, 'test'))
    elif eval_config.DATASET_TYPE == 'scannet_val':
        scenes = sorted(scenelists.get_scenes_scannet(config.SCANNET_DIR, 'val'))
    elif eval_config.DATASET_TYPE == 'icl-nuim':
        scenes = sorted(scenelists.get_scenes_icl_nuim(config.ICL_NUIM_DIR))
    elif eval_config.DATASET_TYPE == 'tum-rgbd':
        scenes = sorted(scenelists.get_scenes_tum_rgbd(config.TUM_RGBD_DIR))
    else:
        raise NotImplementedError

    selector = frameselector.NextPoseDistSelector(eval_config.PDIST, 20)
    n_src_on_either_side = eval_config.N_SRC_ON_EITHER_SIDE
    
    dset = dataset.Dataset(scenes, selector, None, (480, 640), augment=False, n_src_on_either_side=n_src_on_either_side,
                           **dset_kwargs)

    net = net.to(DEVICE)
    net.eval()

    start_idx = 0
    for j, scene in enumerate(scenes[start_idx:]):
        scene_name = os.path.basename(scene)
        print('{} / {}: {}'.format(j + 1 + start_idx, len(scenes), scene_name))
        scene_save_dir = os.path.join(save_dir, 'scenes', scene_name)
        if not os.path.exists(scene_save_dir):
            os.makedirs(scene_save_dir)
        pred_file_path = os.path.join(scene_save_dir, 'preds.npz')

        # make predictions
        if not os.path.exists(pred_file_path) or overwrite:
            batch, _, _, img_idx = dset.get(j+start_idx, return_verbose=True, seed_idx=0)
            batch.__setattr__('images_batch', torch.zeros(batch.images.shape[0], dtype=torch.long))
            batch = batch.to(DEVICE)
            ref_idx = torch.unique(batch.ref_src_edges[0]).detach().cpu().numpy()

            if depth:
                depth_preds, init_prob, final_prob = pred_func(batch, scene, dset, net)
            else:
                mesh = pred_func(batch, scene, dset, net)
                o3d.io.write_triangle_mesh(os.path.join(scene_save_dir, 'mesh.ply'), mesh)

                # render depth predictions from mesh
                poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2).detach().cpu().numpy()
                eye_row = np.repeat(np.array([[[0., 0., 0., 1.]]]), poses.shape[0], axis=0)
                poses = np.concatenate((poses, eye_row), axis=1)
                K = batch.K.detach().cpu().numpy()

                depth_preds = meshtodepth.process_scene(mesh, poses[ref_idx], K[ref_idx])
                init_prob = None
                final_prob = None

            # convert K to size of depth images
            old_h, old_w = batch.images.shape[-2:]
            new_h, new_w = depth_preds.shape[-2:]
            x_fact = float(new_w) / float(old_w)
            y_fact = float(new_h) / float(old_h)
            K = batch.K[ref_idx]
            K[:, 0, :] *= x_fact
            K[:, 1, :] *= y_fact

            # save depth prediction results to preds.npz file, to be used for calculating 2D/3D metrics
            preds = dict(
                scene=os.path.basename(scene),
                depth_preds=depth_preds,
                rotmats=batch.rotmats[ref_idx].detach().cpu().numpy(),
                tvecs=batch.tvecs[ref_idx].detach().cpu().numpy(),
                K=K.detach().cpu().numpy(),
                img_idx=img_idx[ref_idx],       # stores indices of all reference images
            )
            # save probability maps for fmvs and pmvs to be used in point cloud fusion
            if init_prob is not None:
                preds['init_prob'] = init_prob
            if final_prob is not None:
                preds['final_prob'] = final_prob

            np.savez(
                pred_file_path,
                **preds
            )

        # calculated 2D and 3D metrics from the network predictions in preds.npz file
        processresults.process_scene_2d_metrics(scene, scene_save_dir, overwrite)
        torch.cuda.empty_cache()
        if depth:
            processresults.process_depth_3d_metrics(
                scene, scene_save_dir, eval_config.Z_THRESH, eval_config.RUN_TSDF_FUSION, eval_config.RUN_PCFUSION,
                overwrite)
        else:
            processresults.process_volume_3d_metrics(scene, scene_save_dir, overwrite)

    processresults.calc_avg_metrics(save_dir)   # calculate and save aggregated metrics
