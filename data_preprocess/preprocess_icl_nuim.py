from mv3d import config
from mv3d.dsets.scenelists import get_scenes_icl_nuim
from mv3d.dsets import frameselector
from mv3d.dsets import dataset
from mv3d.eval import tsdf_atlas
import torch
import os
import numpy as np
import glob
import tqdm
import json
import cv2
from scipy.spatial.transform.rotation import Rotation
import open3d as o3d


# For K, https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
K = np.array([[481.20,	0,	319.50],
              [0, -480.00,	239.50],
              [0,	0,	1]])
IMG_BATCH = 20

# Volumetric-based 3D options
VOX_RES = 0.02
VOL_PRCNT = .995
VOL_MARGIN = 1.5
TRUNC_RATIO = 3


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


def fix_pose(P):
    theta = (1. / 2.) * np.pi
    rotvec = np.array([theta, 0, 0], dtype=np.float32)
    R, _ = cv2.Rodrigues(rotvec)
    R = R.astype(np.float32)
    P_fix = np.eye(4)
    P_fix[:3, :3] = R
    P = P_fix @ P
    return P


def generate_gt_mesh(scene):
    # get frames for trimming mesh
    selector = frameselector.EveryNthSelector(1)
    dset = dataset.Dataset([scene, ], selector, IMG_BATCH, (480, 640), (480, 640), False, scale_rgb=1.,
                           mean_rgb=[0., 0., 0.], std_rgb=[1., 1., 1.], crop=False, n_src_on_either_side=0)
    n_imgs = len(json.load(open(os.path.join(scene, 'info.json'), 'r'))['frames'])
    n_batches = (n_imgs - 1) // IMG_BATCH + 1

    # first, run through to get vol bounds
    origin = None
    vol_max = None
    for i in range(n_batches):
        batch, _, _, img_idx = dset.get(0, seed_idx=i * IMG_BATCH, return_verbose=True)
        poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2)
        eye_row = torch.tensor([[[0., 0., 0., 1.]]]).repeat(poses.shape[0], 1, 1)
        poses = torch.cat((poses, eye_row), dim=1)

        P = get_projection_matrices(batch.K, poses).cuda()
        pts = depth_projection_batched(batch.depth_images.cuda(), P)
        pts = pts.reshape(-1, 3)
        pts = pts[~torch.any(torch.isnan(pts), dim=1)]  # avoid any div/0s in projection step
        pts = pts.cpu().numpy()
        if pts.shape[0] == 0:  # avoid any fully 0 depth maps from cut off reconstruction
            continue

        origin_batch = torch.as_tensor(np.quantile(pts, 1 - VOL_PRCNT, axis=0) - VOL_MARGIN).float()
        vol_max_batch = torch.as_tensor(np.quantile(pts, VOL_PRCNT, axis=0) + VOL_MARGIN).float()

        if origin is None:
            origin = origin_batch
        else:
            origin = torch.min(torch.stack((origin, origin_batch), dim=0), dim=0)[0]

        if vol_max is None:
            vol_max = vol_max_batch
        else:
            vol_max = torch.max(torch.stack((vol_max, vol_max_batch), dim=0), dim=0)[0]

    # use top and bottom vol_prcnt of points plus vol_margin
    vol_dim = ((vol_max - origin) / VOX_RES).int().tolist()

    # re-run through dset for tsdf fusion step
    tsdf_fusion = tsdf_atlas.TSDFFusion(vol_dim, VOX_RES, origin, TRUNC_RATIO, torch.device('cuda'),
                                        label=False)
    for i in range(n_batches):
        batch, _, _, img_idx = dset.get(0, seed_idx=i * IMG_BATCH, return_verbose=True)

        poses = torch.cat((batch.rotmats, batch.tvecs.unsqueeze(2)), dim=2)
        eye_row = torch.tensor([[[0., 0., 0., 1.]]]).repeat(poses.shape[0], 1, 1)
        poses = torch.cat((poses, eye_row), dim=1)

        P = get_projection_matrices(batch.K, poses).cuda()
        for i in range(batch.depth_images.shape[0]):
            projection = P[i]
            image = batch.images[i].cuda()
            depth = batch.depth_images[i]
            tsdf_fusion.integrate(projection, depth, image)

    tsdf = tsdf_fusion.get_tsdf()
    trimmed_mesh = tsdf.get_mesh()
    return trimmed_mesh


def process_scene(scene, overwrite_depth=False, overwrite_mesh=False):
    scene_name = os.path.basename(scene)
    process_depth_dir = os.path.join(scene, 'depth_processed')
    gt_mesh_file = os.path.join(scene, 'gt_mesh.ply')
    if not os.path.exists(process_depth_dir):
        os.makedirs(process_depth_dir)
    data = {
        'scene': scene_name,
        'path': scene,
        'intrinsics': K.tolist(),
        'gt_mesh': gt_mesh_file,
        'frames': []
    }

    raw_frames_info = open(os.path.join(scene, 'associations.txt'), 'r').readlines()
    raw_poses_info = open(glob.glob(os.path.join(scene, '*.gt.freiburg'))[0], 'r').readlines()
    poses_dict = {}
    for raw_pose_info in raw_poses_info:
        split = raw_pose_info.strip().split(' ')
        poses_dict[split[0]] = np.asarray([float(f) for f in split[1:]])

    for i, raw_frame_info in enumerate(raw_frames_info):
        split = raw_frame_info.strip().split(' ')

        if split[0] not in poses_dict.keys():
            continue

        pose_raw = poses_dict[split[0]]
        depth_filepath_src = os.path.join(scene, split[1])
        depth_filepath_dst = os.path.join(process_depth_dir, os.path.basename(depth_filepath_src))
        color_filepath = os.path.join(scene, split[3])

        rotmat = Rotation.from_quat(pose_raw[3:]).as_matrix()
        tvec = np.reshape(pose_raw[:3], (3, 1))
        eye_row = np.array([[0., 0., 0., 1.]])
        P = np.concatenate((rotmat, tvec), axis=1)
        P = np.concatenate((P, eye_row), axis=0)
        P = fix_pose(P)

        if not np.all(np.isfinite(P)):  # skip invalid poses
            continue

        if overwrite_depth or not os.path.exists(depth_filepath_dst):
            depth = cv2.imread(depth_filepath_src, cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float) / 5_000.
            depth[np.isnan(depth)] = 0.
            depth[np.isinf(depth)] = 0.
            depth = (depth * 1000).astype(np.uint16)
            cv2.imwrite(depth_filepath_dst, depth)

        frame = {
            'filename_color': color_filepath,
            'filename_depth': depth_filepath_dst,
            'pose': P.tolist()
        }
        data['frames'].append(frame)
    json.dump(data, open(os.path.join(scene, 'info.json'), 'w'))

    if not os.path.exists(gt_mesh_file) or overwrite_mesh:
        gt_mesh = generate_gt_mesh(scene)
        o3d.io.write_triangle_mesh(gt_mesh_file, gt_mesh)

    return


if __name__ == '__main__':
    scenes = get_scenes_icl_nuim(config.ICL_NUIM_DIR)
    for scene in tqdm.tqdm(scenes):
        process_scene(scene)
