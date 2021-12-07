from mv3d.eval import pointcloudfusion_gipuma
from mv3d.eval import pointcloudfusion_custom
from mv3d.eval import metricfunctions
from mv3d.eval import meshtodepth
from mv3d.eval import tsdf_atlas
from mv3d.eval import config as eval_config
from mv3d.dsets import dataset, frameselector
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import cv2
import open3d as o3d
import glob
import tqdm


def get_projection_matrices(K, poses):
    n_imgs = poses.shape[0]
    zeros = torch.zeros((n_imgs, 3, 1)).type_as(K)  # append a column of 0s to camera intrinsics
    K = torch.cat((K, zeros), dim=2)
    P = torch.bmm(K, poses)
    return P


def depth_projection_batched(depth, P):
    h, w = depth.shape[-2:]
    n_imgs = depth.shape[0]

    pts_y = torch.arange(h).type_as(depth)
    pts_x = torch.arange(w).type_as(depth)
    pts_yy, pts_xx = torch.meshgrid(pts_y, pts_x)
    pts_zz = torch.ones_like(pts_xx)

    pts_H = torch.stack((pts_xx, pts_yy, pts_zz), dim=0)
    pts_H = pts_H.unsqueeze(0).repeat(n_imgs, 1, 1, 1)

    pts_ww = 1. / depth.unsqueeze(1)
    pts_H = torch.cat((pts_H, pts_ww), dim=1)

    # add [0, 0, 0, 1] row to projection matrix to make 4x4
    eye_row = torch.tensor([[0, 0, 0, 1]]).unsqueeze(0).repeat(n_imgs, 1, 1).type_as(P)
    P = torch.cat((P, eye_row), dim=1)
    P_inv = P.inverse()

    pts_H = torch.bmm(P_inv, pts_H.view(n_imgs, 4, h*w))
    pts = pts_H[:, :3] / pts_H[:, 3:]

    return pts.transpose(2, 1)


def load_gt_depth(img_idx, scene):
    scene_info = json.load(open(os.path.join(scene, 'info.json'), 'r'))
    depth_gt = [cv2.imread(scene_info['frames'][idx]['filename_depth'], cv2.IMREAD_ANYDEPTH).astype(np.float) / 1000.
                for idx in img_idx]
    depth_gt = np.stack(depth_gt, axis=0)
    # depth_gt = torch.from_numpy(np.stack(depth_gt, axis=0))
    return depth_gt


def load_images(img_idx, scene):
    scene_info = json.load(open(os.path.join(scene, 'info.json'), 'r'))
    images = [cv2.cvtColor(cv2.imread(scene_info['frames'][idx]['filename_color']), cv2.COLOR_BGR2RGB)
              for idx in img_idx]
    # images = [cv2.imread(scene_info['frames'][idx]['filename_color']) for idx in img_idx]
    images = np.stack(images, axis=0)
    return images


def trim_mesh(mesh, scene):
    # get frames for trimming mesh
    selector = frameselector.EveryNthSelector(1)
    dset = dataset.Dataset([scene, ], selector, eval_config.IMG_BATCH, (480, 640), (480, 640), False, scale_rgb=1.,
                           mean_rgb=[0., 0., 0.], std_rgb=[1., 1., 1.], crop=False, n_src_on_either_side=0)
    scene_info = json.load(open(os.path.join(scene, 'info.json'), 'r'))
    n_imgs = len(scene_info['frames'])
    n_batches = (n_imgs - 1) // eval_config.IMG_BATCH + 1

    # first, run through to get vol bounds
    print('Calculating volume bounds for trimmed mesh...')
    origin = None
    vol_max = None
    for i in tqdm.tqdm(range(n_batches)):
        batch, _, _, img_idx = dset.get(0, seed_idx=i * eval_config.IMG_BATCH, return_verbose=True)

        poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2)
        eye_row = torch.tensor([[[0., 0., 0., 1.]]]).repeat(poses.shape[0], 1, 1)
        poses = torch.cat((poses, eye_row), dim=1)

        depth_preds = meshtodepth.process_scene(mesh, poses.numpy(), batch.K.numpy())
        depth_preds = torch.from_numpy(depth_preds).float().cuda()

        P = get_projection_matrices(batch.K, poses).cuda()
        pts = depth_projection_batched(depth_preds, P)
        pts = pts.reshape(-1, 3)
        pts = pts[~torch.any(torch.isnan(pts), dim=1)]  # avoid any div/0s in projection step
        pts = pts.cpu().numpy()
        if pts.shape[0] == 0:  # avoid any fully 0 depth maps from cut off reconstruction
            continue

        origin_batch = torch.as_tensor(np.quantile(pts, 1 - eval_config.VOL_PRCNT, axis=0)
                                       - eval_config.VOL_MARGIN).float()
        vol_max_batch = torch.as_tensor(np.quantile(pts, eval_config.VOL_PRCNT, axis=0)
                                        + eval_config.VOL_MARGIN).float()

        if origin is None:
            origin = origin_batch
        else:
            origin = torch.min(torch.stack((origin, origin_batch), dim=0), dim=0)[0]

        if vol_max is None:
            vol_max = vol_max_batch
        else:
            vol_max = torch.max(torch.stack((vol_max, vol_max_batch), dim=0), dim=0)[0]

    # use top and bottom vol_prcnt of points plus vol_margin
    vol_dim = ((vol_max - origin) / eval_config.VOX_RES).int().tolist()

    print('Running tsdf fusion...')
    # re-run through dset for tsdf fusion step
    tsdf_fusion = tsdf_atlas.TSDFFusion(vol_dim, eval_config.VOX_RES, origin, eval_config.TRUNC_RATIO,
                                        torch.device('cuda'), label=False)
    for i in tqdm.tqdm(range(n_batches)):
        batch, _, _, img_idx = dset.get(0, seed_idx=i * eval_config.IMG_BATCH, return_verbose=True)

        poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2)
        eye_row = torch.tensor([[[0., 0., 0., 1.]]]).repeat(poses.shape[0], 1, 1)
        poses = torch.cat((poses, eye_row), dim=1)

        poses_np = poses.numpy()
        K_np = batch.K.numpy()
        depth_preds = meshtodepth.process_scene(mesh, poses_np, K_np)
        if eval_config.MASK_USING_GT_MESH:  # mask out where gt is missing
            gt_mesh = o3d.io.read_triangle_mesh(scene_info['gt_mesh'])
            depth_gt_reproj = meshtodepth.process_scene(gt_mesh, poses_np, K_np)
            depth_preds = np.where(depth_gt_reproj == 0., 0., depth_preds)

        depth_preds = torch.from_numpy(depth_preds).float().cuda()

        P = get_projection_matrices(batch.K, poses).cuda()
        for i in range(depth_preds.shape[0]):
            projection = P[i]
            image = batch.images[i].cuda()
            depth = depth_preds[i]
            tsdf_fusion.integrate(projection, depth, image)

    tsdf = tsdf_fusion.get_tsdf()
    trimmed_mesh = tsdf.get_mesh()
    return trimmed_mesh


def process_scene_2d_metrics(scene, scene_save_dir, overwrite=False):
    pred_file_path = os.path.join(scene_save_dir, 'preds.npz')
    metrics_file_path = os.path.join(scene_save_dir, 'metrics_2d.json')
    if os.path.exists(metrics_file_path) and not overwrite:
        return
    data = np.load(pred_file_path)
    img_idx = data['img_idx']
    depth_gt = torch.from_numpy(load_gt_depth(img_idx, scene)).cuda()
    depth_preds = torch.from_numpy(data['depth_preds']).cuda()
    depth_pred_lg = F.interpolate(depth_preds.unsqueeze(1), depth_gt.shape[-2:], mode='nearest').squeeze(1)
    valid = (depth_pred_lg != 0.) & (~torch.isinf(depth_pred_lg))
    metrics = metricfunctions.calc_2d_depth_metrics_batched(depth_pred_lg, depth_gt, pred_valid=valid,
                                                            batch_size=eval_config.IMG_BATCH)
    metrics['n'] = depth_preds.shape[0]
    json.dump(metrics, open(metrics_file_path, 'w'))
    print(metrics)
    return


def process_volume_3d_metrics(scene, scene_save_dir, overwrite=False):

    # look for mesh.ply file and metrics_3d.json file
    metrics_filepath = os.path.join(
        scene_save_dir, 'metrics_3d_masked.json' if eval_config.MASK_USING_GT_MESH else 'metrics_3d.json')
    mesh_filepath = os.path.join(scene_save_dir, 'mesh.ply')
    trimmed_mesh_filepath = os.path.join(
        scene_save_dir, 'trimmed_mesh_masked.ply' if eval_config.MASK_USING_GT_MESH else 'trimmed_mesh.ply')
    if not os.path.exists(mesh_filepath):
        raise FileNotFoundError
    if os.path.exists(metrics_filepath) and not overwrite:
        return
    mesh = o3d.io.read_triangle_mesh(mesh_filepath)

    trimmed_mesh = trim_mesh(mesh, scene)

    print('Saving and calculating mesh metrics...')
    o3d.io.write_triangle_mesh(trimmed_mesh_filepath, trimmed_mesh)
    pcd_mesh = o3d.io.read_point_cloud(trimmed_mesh_filepath)
    pcd_mesh = pcd_mesh.voxel_down_sample(eval_config.VOXEL_DOWNSAMPLE)

    scene_info = json.load(open(os.path.join(scene, 'info.json'), 'r'))
    pcd_gt = o3d.io.read_point_cloud(scene_info['gt_mesh'])
    pcd_gt = pcd_gt.voxel_down_sample(eval_config.VOXEL_DOWNSAMPLE)

    metrics_mesh = metricfunctions.eval_mesh(pcd_mesh, pcd_gt, eval_config.DIST_THRESH)
    print(metrics_mesh)
    json.dump(metrics_mesh, open(metrics_filepath, 'w'))
    return


def process_depth_3d_metrics(scene, scene_save_dir, z_thresh, run_tsdf=False, run_pcfusion=True,
                             overwrite=False):
    pred_file_path = os.path.join(scene_save_dir, 'preds.npz')
    scene_info = json.load(open(os.path.join(scene, 'info.json'), 'r'))

    # generate filenames
    pcd_filename = 'fused_{:.3f}_{}v'.format(z_thresh, eval_config.N_CONSISTENT_THRESH)
    metrics_filename = 'metrics_3d_{:.3f}_{}v'.format(z_thresh, eval_config.N_CONSISTENT_THRESH)
    if eval_config.MASK_USING_GT_MESH:
        pcd_filename += '_masked'
        metrics_filename += '_masked'
    pcd_filepath = os.path.join(scene_save_dir, '{}.ply'.format(pcd_filename))
    metrics_filepath = os.path.join(scene_save_dir, '{}.json'.format(metrics_filename))

    if (not os.path.exists(pcd_filepath) or overwrite) and run_pcfusion:
        data = np.load(pred_file_path)
        depth_preds = data['depth_preds']
        init_prob = data['init_prob'] if 'init_prob' in data.keys() else None
        final_prob = data['final_prob'] if 'final_prob' in data.keys() else None
        n_imgs = depth_preds.shape[0]
        rotmats = data['rotmats']
        tvecs = data['tvecs']
        K = data['K']
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], n_imgs, axis=0)
        poses[:, :3, :3] = rotmats
        poses[:, :3, 3] = tvecs
        depth_gt = load_gt_depth(data['img_idx'], scene)
        images = load_images(data['img_idx'], scene)

        if init_prob is not None:
            for i in range(n_imgs):
                p = init_prob[i]
                if p.shape[-2:] != depth_preds.shape[-2:]:
                    p = cv2.resize(p, (depth_preds.shape[2], depth_preds.shape[1]), interpolation=cv2.INTER_LANCZOS4)
                depth_preds[i] = np.where(p > 0.2, depth_preds[i], 0.)
        if final_prob is not None:
            for i in range(n_imgs):
                p = final_prob[i]
                if p.shape[-2:] != depth_preds.shape[-2:]:
                    p = cv2.resize(p, (depth_preds.shape[2], depth_preds.shape[1]), interpolation=cv2.INTER_LANCZOS4)
                depth_preds[i] = np.where(p > 0.1, depth_preds[i], 0.)

        # decide whether to use fusibile based on num of images (get segfault at >1024 images)
        use_fusibile = n_imgs < 1024 and os.path.exists(eval_config.FUSIBILE_EXE_PATH)

        # fusibile expects BGR images
        if use_fusibile:
            images = images[..., [2, 1, 0]]

        # convert depth preds to (480, 640) if necessary
        if depth_preds.shape[-2:] != depth_gt.shape[-2:]:
            x_fact = depth_gt.shape[-1] / float(depth_preds.shape[-1])
            y_fact = depth_gt.shape[-2] / float(depth_preds.shape[-2])
            depth_preds = F.interpolate(torch.from_numpy(depth_preds).unsqueeze(1), depth_gt.shape[-2:], mode='nearest')\
                .squeeze(1)\
                .numpy()
            K[:, 0, :] *= x_fact
            K[:, 1, :] *= y_fact

        # mask out depth predictions where ground truth reconstruction does not exist
        if eval_config.MASK_USING_GT_MESH:
            gt_mesh = o3d.io.read_triangle_mesh(scene_info['gt_mesh'])
            depth_gt_reproj = meshtodepth.process_scene(gt_mesh, poses, K)
            depth_preds = np.where(depth_gt_reproj == 0., 0., depth_preds)

        print('Running point cloud fusion{}...'.format(' with fusibile' if use_fusibile else ' with slow pytorch fusion'))
        print('Reprojection thresh: {:.3f}'.format(z_thresh))
        print('Fusing...')
        if use_fusibile:
            pcd_pred = pointcloudfusion_gipuma.process_scene(depth_preds, images, poses, K,
                                                             eval_config.FUSIBILE_EXE_PATH, z_thresh,
                                                             eval_config.N_CONSISTENT_THRESH)
        else:
            fused_pts, fused_rgb, _ = pointcloudfusion_custom.process_scene(
                torch.from_numpy(depth_preds), torch.from_numpy(images), torch.from_numpy(poses),
                torch.from_numpy(K), z_thresh, eval_config.N_CONSISTENT_THRESH)
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(fused_pts)
            pcd_pred.colors = o3d.utility.Vector3dVector(fused_rgb / 255.)

        print('Saving and calculating metrics...')
        pcd_pred = pcd_pred.voxel_down_sample(eval_config.VOXEL_DOWNSAMPLE)
        o3d.io.write_point_cloud(pcd_filepath, pcd_pred)

        pcd_gt = o3d.io.read_point_cloud(scene_info['gt_mesh'])
        pcd_gt = pcd_gt.voxel_down_sample(eval_config.VOXEL_DOWNSAMPLE)

        metrics_3d = metricfunctions.eval_mesh(pcd_pred, pcd_gt, eval_config.DIST_THRESH)
        metrics_3d['n'] = depth_preds.shape[0]
        print(metrics_3d)

        o3d.io.write_point_cloud(pcd_filepath, pcd_pred)
        json.dump(metrics_3d, open(metrics_filepath, 'w'))

    if run_tsdf:
        mesh_filename = 'tsdf_mesh'
        metrics_mesh_filename = 'metrics_tsdf'
        if eval_config.MASK_USING_GT_MESH:
            mesh_filename += '_masked'
            metrics_mesh_filename += '_masked'
        mesh_filepath = os.path.join(scene_save_dir, '{}.ply'.format(mesh_filename))
        metrics_mesh_filepath = os.path.join(scene_save_dir, '{}.json'.format(metrics_mesh_filename))
        if not os.path.exists(mesh_filepath) or overwrite:
            data = np.load(pred_file_path)
            depth_preds = data['depth_preds']
            n_imgs = depth_preds.shape[0]
            rotmats = data['rotmats']
            tvecs = data['tvecs']
            K = data['K']
            poses = np.repeat(np.eye(4, dtype=np.float32)[None], n_imgs, axis=0)
            poses[:, :3, :3] = rotmats
            poses[:, :3, 3] = tvecs
            images = load_images(data['img_idx'], scene)

            K = torch.from_numpy(K)
            poses = torch.from_numpy(poses)
            images = torch.from_numpy(images[..., [2, 1, 0]]).permute(0, 3, 1, 2).float()
            images = F.interpolate(images, depth_preds.shape[-2:], mode='bilinear')

            n_batches = (n_imgs - 1) // eval_config.IMG_BATCH + 1

            # first, run through to get vol bounds
            print('Calculating volume bounds for trimmed mesh...')
            origin = None
            vol_max = None
            for i in tqdm.tqdm(range(n_batches)):
                start_idx = i * eval_config.IMG_BATCH
                end_idx = (i+1)*eval_config.IMG_BATCH
                depth_preds_batch = torch.from_numpy(depth_preds[start_idx:end_idx]).float().cuda()

                P = get_projection_matrices(K[start_idx: end_idx], poses[start_idx: end_idx]).cuda()
                pts = depth_projection_batched(depth_preds_batch, P)
                pts = pts.reshape(-1, 3)
                pts = pts[~torch.any(torch.isnan(pts), dim=1)]  # avoid any div/0s in projection step
                pts = pts.cpu().numpy()
                if pts.shape[0] == 0:  # avoid any fully 0 depth maps from cut off reconstruction
                    continue

                origin_batch = torch.as_tensor(np.quantile(pts, 1 - eval_config.VOL_PRCNT, axis=0)
                                               - eval_config.VOL_MARGIN).float()
                vol_max_batch = torch.as_tensor(np.quantile(pts, eval_config.VOL_PRCNT, axis=0)
                                                + eval_config.VOL_MARGIN).float()

                if origin is None:
                    origin = origin_batch
                else:
                    origin = torch.min(torch.stack((origin, origin_batch), dim=0), dim=0)[0]

                if vol_max is None:
                    vol_max = vol_max_batch
                else:
                    vol_max = torch.max(torch.stack((vol_max, vol_max_batch), dim=0), dim=0)[0]

            # use top and bottom vol_prcnt of points plus vol_margin
            vol_dim = ((vol_max - origin) / eval_config.VOX_RES).int().tolist()

            print('Running tsdf fusion...')
            # re-run through dset for tsdf fusion step
            tsdf_fusion = tsdf_atlas.TSDFFusion(vol_dim, eval_config.VOX_RES, origin, eval_config.TRUNC_RATIO,
                                                torch.device('cuda'), label=False)
            gt_mesh = o3d.io.read_triangle_mesh(scene_info['gt_mesh'])
            for i in tqdm.tqdm(range(n_batches)):
                start_idx = i * eval_config.IMG_BATCH
                end_idx = (i+1)*eval_config.IMG_BATCH
                depth_preds_batch = depth_preds[start_idx: end_idx]
                if eval_config.MASK_USING_GT_MESH:  # mask out where gt is missing
                    depth_gt_reproj = meshtodepth.process_scene(gt_mesh, poses[start_idx: end_idx],
                                                                K[start_idx: end_idx], depth_preds.shape[-2:])
                    depth_preds_batch = np.where(depth_gt_reproj == 0., 0., depth_preds_batch)

                depth_preds_batch = torch.from_numpy(depth_preds_batch).float().cuda()

                P = get_projection_matrices(K[start_idx: end_idx], poses[start_idx: end_idx]).cuda()
                for i in range(depth_preds_batch.shape[0]):
                    projection = P[i]
                    image = images[start_idx: end_idx][i].cuda()
                    depth = depth_preds_batch[i]
                    tsdf_fusion.integrate(projection, depth, image)

            tsdf = tsdf_fusion.get_tsdf()
            mesh = tsdf.get_mesh()

            print('Saving and calculating mesh metrics...')
            o3d.io.write_triangle_mesh(mesh_filepath, mesh)
            pcd_mesh = o3d.io.read_point_cloud(mesh_filepath)
            pcd_mesh = pcd_mesh.voxel_down_sample(eval_config.VOXEL_DOWNSAMPLE)

            pcd_gt = o3d.io.read_point_cloud(scene_info['gt_mesh'])
            pcd_gt = pcd_gt.voxel_down_sample(eval_config.VOXEL_DOWNSAMPLE)

            metrics_mesh = metricfunctions.eval_mesh(pcd_mesh, pcd_gt, eval_config.DIST_THRESH)
            metrics_mesh['n'] = depth_preds.shape[0]
            print(metrics_mesh)

            json.dump(metrics_mesh, open(metrics_mesh_filepath, 'w'))

    return


def calc_avg_metrics(save_dir):
    scenes_parent_dir = os.path.join(save_dir, 'scenes')
    scenes_save_dirs = [os.path.join(scenes_parent_dir, s) for s in sorted(os.listdir(scenes_parent_dir))]

    # look in first scene dir and determine what metrics files exist
    metrics_filenames = [os.path.basename(f) for f in glob.glob(os.path.join(scenes_save_dirs[0], 'metrics*.json'))]

    for metrics_filename in metrics_filenames:
        all_metrics = []
        for scene in os.listdir(scenes_parent_dir):
            metrics_filepath = os.path.join(scenes_parent_dir, scene, metrics_filename)
            if os.path.exists(metrics_filepath):
                all_metrics.append(json.load(open(metrics_filepath, 'r')))
        avg_metrics = {}
        n_sum = np.sum([m['n'] for m in all_metrics]) if 'n' in all_metrics[0].keys() else 1.
        for k in all_metrics[0].keys():
            if k != 'n':
                if k in ['acc', 'comp', 'prec', 'recal', 'fscore']:
                    avg_metrics[k] = np.mean([m[k] for m in all_metrics])
                else:
                    avg_metrics[k] = float(np.sum([m['n']*m[k] for m in all_metrics])) / n_sum
        print('-'*10)
        print(metrics_filename)
        print(avg_metrics)
        json.dump(avg_metrics, open(os.path.join(save_dir, metrics_filename), 'w'))
    return
