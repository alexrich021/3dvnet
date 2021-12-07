from mv3d import config
from mv3d.dsets import scenelists
from mv3d.dsets import frameselector
from mv3d.dsets import dataset
from mv3d.eval import tsdf_atlas
import os
import numpy as np
import glob
import tqdm
import json
import cv2
import torch
from scipy.spatial.transform.rotation import Rotation
import open3d as o3d


K = np.array([[525.0, 0.0, 320.0],
              [0.0, 525.0, 240.0],
              [0.0, 0.0, 1.0]])
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


def generate_gt_mesh(scene, max_depth=None):
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

        # mask out large depth values if necessary
        if max_depth is not None:
            batch.depth_images[batch.depth_images > 4.] = 0.

        P = get_projection_matrices(batch.K, poses).cuda()
        pts = depth_projection_batched(batch.depth_images.cuda(), P)
        pts = pts.reshape(-1, 3)
        pts = pts[~torch.any(torch.isnan(pts), dim=1)]  # avoid any div/0s in projection step
        pts = pts.cpu().numpy()
        if pts.shape[0] == 0:  # avoid any fully 0 depth maps from cut off reconstruction
            continue

        # use top and bottom vol_prcnt of points plus vol_margin
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


def get_closest_index(target_timestamp, other_timestamps):
    differences = np.abs(other_timestamps - target_timestamp)
    return np.argmin(differences)


def process_scene(scene, overwrite_depth=False, overwrite_mesh=False):
    depth_out_dir = os.path.join(scene, 'depth_processed')
    if not os.path.exists(depth_out_dir):
        os.makedirs(depth_out_dir)
    scene_name = os.path.basename(scene)
    gt_mesh_file = os.path.join(scene, 'gt_mesh.ply')

    data = {
        'scene': scene_name,
        'path': scene,
        'intrinsics': K.tolist(),
        'gt_mesh': gt_mesh_file,
        'frames': []
    }

    image_filenames = sorted(glob.glob(os.path.join(scene, 'rgb', '*.png')))
    image_timestamps = np.loadtxt(os.path.join(scene, 'rgb.txt'), usecols=0)

    depth_filenames = sorted(glob.glob(os.path.join(scene, 'depth', '*.png')))
    depth_timestamps = np.loadtxt(os.path.join(scene, 'depth.txt'), usecols=0)

    poses_with_quat = np.loadtxt(os.path.join(scene, 'groundtruth.txt'))
    pose_timestamps = poses_with_quat[:, 0]
    pose_locations = poses_with_quat[:, 1:4]
    pose_quaternions = poses_with_quat[:, 4:]

    for i in range(len(depth_filenames)):
        depth_timestamp = depth_timestamps[i]

        pose_index = get_closest_index(depth_timestamp, pose_timestamps)
        image_index = get_closest_index(depth_timestamp, image_timestamps)

        depth_filename_src = depth_filenames[i]
        depth_filename_dst = os.path.join(depth_out_dir, os.path.basename(depth_filename_src))
        image_filename = image_filenames[image_index]
        pose_location = pose_locations[pose_index]
        pose_quaternion = pose_quaternions[pose_index]
        rot = Rotation.from_quat(pose_quaternion).as_matrix()
        pose = np.eye(4)
        pose[0:3, 0:3] = rot
        pose[0:3, 3] = pose_location

        if not os.path.exists(depth_filename_dst) or overwrite_depth:
            depth = cv2.imread(depth_filename_src, -1).astype(np.float32) / 5000.
            depth[np.isnan(depth)] = 0.
            depth[np.isinf(depth)] = 0.
            depth = (depth * 1000.).astype(np.uint16)
            cv2.imwrite(depth_filename_dst, depth)

        frame = {
            'filename_color': image_filename,
            'filename_depth': depth_filename_dst,
            'pose': pose.tolist()
        }
        data['frames'].append(frame)
    json.dump(data, open(os.path.join(scene, 'info.json'), 'w'))

    if not os.path.exists(gt_mesh_file) or overwrite_mesh:
        gt_mesh = generate_gt_mesh(scene)
        o3d.io.write_triangle_mesh(gt_mesh_file, gt_mesh)
    return


if __name__ == '__main__':
    scenes = scenelists.get_scenes_tum_rgbd(config.TUM_RGBD_DIR)
    for scene in tqdm.tqdm(scenes):
        process_scene(scene)
