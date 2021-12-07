"""
Copyright 2019, Yao Yao, HKUST.
Edited by Alex Rich
Convert MV3D output to Gipuma format for post-processing.
"""

import os
import glob
import shutil
from struct import *

import cv2
import numpy as np
import open3d as o3d


def read_gipuma_dmb(path):
    '''read Gipuma .dmb format image'''

    with open(path, "rb") as fid:
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]

        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_gipuma_dmb(path, image):
    '''write Gipuma .dmb format image'''

    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()

    with open(path, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)
    return


def mv3d_to_gipuma_cam(K, R, t, out_path):
    '''convert mvsnet camera to gipuma camera format'''

    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = t
    K = np.concatenate((K, np.zeros((3, 1))), axis=1)
    P = K @ V

    f = open(out_path, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(P[i, j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()

    return


def fake_gipuma_normal(in_depth_path, out_normal_path):
    depth_image = read_gipuma_dmb(in_depth_path)
    image_shape = np.shape(depth_image)

    normal_image = np.ones_like(depth_image)
    normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1))
    normal_image = np.tile(normal_image, [1, 1, 3])
    normal_image = normal_image / 1.732050808

    mask_image = np.squeeze(np.where(depth_image > 0, 1, 0))
    mask_image = np.reshape(mask_image, (image_shape[0], image_shape[1], 1))
    mask_image = np.tile(mask_image, [1, 1, 3])
    mask_image = np.float32(mask_image)

    normal_image = np.multiply(normal_image, mask_image)
    normal_image = np.float32(normal_image)

    write_gipuma_dmb(out_normal_path, normal_image)
    return


def mv3d_to_gipuma(depth_preds, images, poses, K, tmp_folder):

    rotmats = poses[:, :3, :3]
    tvecs = poses[:, :3, 3]

    gipuma_cam_folder = os.path.join(tmp_folder, 'cams')
    gipuma_image_folder = os.path.join(tmp_folder, 'images')
    if not os.path.isdir(tmp_folder):
        os.mkdir(tmp_folder)
    if not os.path.isdir(gipuma_cam_folder):
        os.mkdir(gipuma_cam_folder)
    if not os.path.isdir(gipuma_image_folder):
        os.mkdir(gipuma_image_folder)

    gipuma_prefix = '2333__'
    n_imgs = depth_preds.shape[0]
    for i in range(n_imgs):
        image_prefix = str(i).zfill(8)
        depth_pred = cv2.resize(depth_preds[i], (640, 480), interpolation=cv2.INTER_NEAREST)

        # convert cameras
        out_cam_file = os.path.join(gipuma_cam_folder, '{}.jpg.P'.format(image_prefix))
        mv3d_to_gipuma_cam(K[i], rotmats[i], tvecs[i], out_cam_file)

        # convert depth maps and fake normal maps
        sub_depth_folder = os.path.join(tmp_folder, gipuma_prefix + image_prefix)
        if not os.path.isdir(sub_depth_folder):
            os.mkdir(sub_depth_folder)
        out_depth_dmb = os.path.join(sub_depth_folder, 'disp.dmb')
        fake_normal_dmb = os.path.join(sub_depth_folder, 'normals.dmb')
        write_gipuma_dmb(out_depth_dmb, depth_pred)
        fake_gipuma_normal(out_depth_dmb, fake_normal_dmb)

        # copy images to gipuma image folder
        out_image_file = os.path.join(gipuma_image_folder, '{}.jpg'.format(image_prefix))
        cv2.imwrite(out_image_file, images[i])


def depth_map_fusion(point_folder, fusibile_exe_path, disp_thresh, num_consistent):
    cam_folder = os.path.join(point_folder, 'cams')
    image_folder = os.path.join(point_folder, 'images')
    depth_min = 0.001
    depth_max = 100000
    normal_thresh = 360

    cmd = fusibile_exe_path
    cmd = cmd + ' -input_folder ' + point_folder + '/'
    cmd = cmd + ' -p_folder ' + cam_folder + '/'
    cmd = cmd + ' -images_folder ' + image_folder + '/'
    cmd = cmd + ' --depth_min=' + str(depth_min)
    cmd = cmd + ' --depth_max=' + str(depth_max)
    cmd = cmd + ' --normal_thresh=' + str(normal_thresh)
    cmd = cmd + ' --disp_thresh=' + str(disp_thresh)
    cmd = cmd + ' --num_consistent=' + str(num_consistent)
    # cmd = cmd + ' --cam_scale=' + str(100.)
    cmd += ' >/dev/null 2>&1'  # suppress console output (comment out for debug)
    # print(cmd)
    os.system(cmd)

    return


def process_scene(depth_preds, images, poses, K, fusibile_exe_path, disp_threshold, num_consistent):
    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    mv3d_to_gipuma(depth_preds, images, poses, K, tmp_dir)
    depth_map_fusion(tmp_dir, fusibile_exe_path, disp_threshold, num_consistent)
    pcd_file = sorted(glob.glob(os.path.join(tmp_dir, 'consistencyCheck-*', 'final3d_model.ply')))[-1]
    pcd = o3d.io.read_point_cloud(pcd_file)
    shutil.rmtree(tmp_dir)
    return pcd
