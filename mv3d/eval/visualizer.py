import open3d as o3d
import os
import numpy as np
import json
import tqdm
from mv3d import config
from mv3d.eval import config as eval_config

res_dir = eval_config.SAVE_DIR
parent_dirs = [         # specify which methods to visualize
    os.path.join(res_dir, '3dvnet', 'scenes'),
    os.path.join(res_dir, 'pmvs_ft', 'scenes'),
    os.path.join(res_dir, 'atlas', 'scenes'),
]
ply_filenames = [       # specify corresponding reconstruction to view for entries in previous list
    'fused_0.010_3v_masked.ply',
    'fused_0.010_3v_masked.ply',
    'trimmed_mesh_masked.ply',
]
metrics_filenames = [   # specify corresponding metrics file to print for entries in previous list
    'metrics_3d_0.010_3v_masked.json',
    'metrics_3d_0.010_3v_masked.json',
    'metrics_3d_masked.json',
]
scene_res_dirs = [[os.path.join(dir, d) for d in sorted(os.listdir(dir))] for dir in parent_dirs]
scenes_dir = os.path.join(config.SCANNET_DIR, 'scans_test')


def create_wireframe_rectangle(xyz_min, xyz_max, color=(0.9, 0.1, 0.1)):
    points = np.concatenate([
        np.array([[xyz_min[0], xyz_min[1], xyz_min[2]]]),
        np.array([[xyz_min[0], xyz_max[1], xyz_min[2]]]),
        np.array([[xyz_max[0], xyz_max[1], xyz_min[2]]]),
        np.array([[xyz_max[0], xyz_min[1], xyz_min[2]]]),
        np.array([[xyz_min[0], xyz_min[1], xyz_max[2]]]),
        np.array([[xyz_min[0], xyz_max[1], xyz_max[2]]]),
        np.array([[xyz_max[0], xyz_max[1], xyz_max[2]]]),
        np.array([[xyz_max[0], xyz_min[1], xyz_max[2]]]),
    ], axis=0)
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7]
    ]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines))
    if color is not None:
        colors = [color for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def print_info(p_idx, s_idx):
    print('{}: {}'.format(parent_dirs[p_idx], ply_filenames[p_idx]))
    metrics_3d_filepath = os.path.join(scene_res_dirs[p_idx][s_idx], metrics_filenames[p_idx])
    metrics_2d_filepath = os.path.join(scene_res_dirs[p_idx][s_idx], 'metrics_2d.json')
    if os.path.exists(metrics_3d_filepath):
        metrics_3d = json.load(open(metrics_3d_filepath, 'r'))
        for k in ['fscore', 'prec', 'recal']:
            print('{}:\t\t{:.3f}'.format(k, metrics_3d[k]))
    else:
        print('3D metrics not found!')
    if os.path.exists(metrics_2d_filepath):
        metrics_2d = json.load(open(metrics_2d_filepath, 'r'))
        for k in ['abs_rel', 'abs_diff']:
            print('{}:\t{:.3f}'.format(k, metrics_2d[k]))
    else:
        print('2D metrics not found!')


def gen_objs(s_idx):

    pcds = []
    for i, ply_filename in tqdm.tqdm(enumerate(ply_filenames), total=len(ply_filenames)):
        filepath = os.path.join(scene_res_dirs[i][s_idx], ply_filename)
        if 'mesh' in ply_filename:
            pcd = o3d.io.read_triangle_mesh(filepath)
            pcd.compute_triangle_normals()
            pcd.compute_vertex_normals()
        else:
            pcd = o3d.io.read_point_cloud(filepath)
        pcds.append(pcd)

    scene = os.path.basename(scene_res_dirs[0][s_idx])
    info = json.load(open(os.path.join(scenes_dir, scene, 'info.json'), 'r'))
    gt_mesh = o3d.io.read_triangle_mesh(info['gt_mesh'])

    # centering all meshes on 0, 0, 0
    gt_pts = np.asarray(gt_mesh.vertices)
    tsdf_min = np.min(gt_pts, axis=0)
    tsdf_max = np.max(gt_pts, axis=0)
    t = tsdf_min + (tsdf_max - tsdf_min) / 2.

    pcds = [pcd.translate(-t) for pcd in pcds]
    gt_mesh = gt_mesh.translate(-t)
    gt_mesh.compute_vertex_normals()

    print('{}:{}'.format(s_idx, scene))

    return pcds, gt_mesh

s_idx = 0
print(s_idx)
p_idx = 0
pcds, gt_mesh = gen_objs(s_idx)
print_info(p_idx, s_idx)
pcds_vis = True
gt_mesh_vis = False


def incr_scene(vis, incr=1):
    global s_idx, pcds, gt_mesh
    toggle_all(vis, 'swap_1')
    s_idx = (s_idx + incr) % len(scene_res_dirs[0])
    pcds, gt_mesh = gen_objs(s_idx)
    toggle_all(vis, 'swap_2')
    print_info(p_idx, s_idx)


def update_obj_state(vis, obj, obj_vis_flag, action='add'):
    if action == 'add' and not obj_vis_flag:
        vis.add_geometry(obj, reset_bounding_box=False)
        return True
    if action == 'remove' and obj_vis_flag:
        vis.remove_geometry(obj, reset_bounding_box=False)
        return False
    if action == 'swap_1':
        if obj_vis_flag:
            vis.remove_geometry(obj, reset_bounding_box=False)
        return obj_vis_flag
    if action == 'swap_2':
        if obj_vis_flag:
            vis.add_geometry(obj, reset_bounding_box=False)
        return obj_vis_flag
    return obj_vis_flag


def toggle_pcds(vis, action='add'):
    global pcds_vis, pcds, p_idx
    pcds_vis = update_obj_state(vis, pcds[p_idx], pcds_vis, action)


def toggle_gt_mesh(vis, action='add'):
    global gt_mesh_vis, gt_mesh
    gt_mesh_vis = update_obj_state(vis, gt_mesh, gt_mesh_vis, action)


def toggle_all(vis, action):
    toggle_pcds(vis, action)
    toggle_gt_mesh(vis, action)


def cycle_pcds(vis, iter=+1):
    global p_idx, pcds, pcds_vis
    pcds_vis = update_obj_state(vis, pcds[p_idx], pcds_vis, 'swap_1')
    p_idx = (p_idx+iter)%len(pcds)
    pcds_vis = update_obj_state(vis, pcds[p_idx], pcds_vis, 'swap_2')
    print_info(p_idx, s_idx)


callbacks = dict()
callbacks[ord('D')] = lambda x: incr_scene(x, +1)
callbacks[ord('X')] = lambda x: incr_scene(x, -1)
callbacks[ord('F')] = lambda x: toggle_gt_mesh(x, 'remove')
callbacks[ord('C')] = lambda x: toggle_gt_mesh(x, 'add')
callbacks[ord('G')] = lambda x: cycle_pcds(x, -1)
callbacks[ord('V')] = lambda x: cycle_pcds(x, +1)
callbacks[ord('J')] = lambda x: toggle_pcds(x, 'remove')
callbacks[ord('N')] = lambda x: toggle_pcds(x, 'add')

o3d.visualization.draw_geometries_with_key_callbacks([pcds[p_idx], ], callbacks)

