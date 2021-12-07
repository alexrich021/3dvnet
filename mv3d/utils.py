import numpy as np
import torch
import cv2
from torch_scatter import scatter
from torchvision import transforms
from torch_geometric.nn import voxel_grid


img_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def freeze_batchnorm(module):
    if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm3d):
        module.eval()
        module.weight.requires_grad = False
        module.bias.requires_grad = False


def random_gravitational_rotation_scannet():
    theta = np.random.uniform(-np.pi, np.pi)
    rotvec = np.array([0, 0, theta], dtype=np.float32)
    R, _ = cv2.Rodrigues(rotvec)
    R = R.astype(np.float32)
    return R


def slice_edges(edges, index_start, index_end, slice_dim=0):
    allowed_index = torch.arange(index_start, index_end, 1, device=edges.device)
    edge_index = torch.abs(allowed_index[None] - edges[slice_dim, :, None]).min(dim=1)[0] == 0
    return edges[:, edge_index]


def voxelize(pts, pts_batch, edge_len):
    bbox_min = pts.min(dim=0)[0]
    bbox_max = pts.max(dim=0)[0]
    grid_size = torch.ceil((bbox_max - bbox_min) / edge_len).long()
    max_grid_idx = grid_size[0] * grid_size[1] * grid_size[2]

    # get 1D voxel idx of pts
    voxel_idx = voxel_grid(pts, pts_batch, edge_len, bbox_min, bbox_max)

    # determine unique set of 1D anchor indices w/ batch information
    anchor_idx, inv_idx = torch.unique(voxel_idx, return_inverse=True)
    anchor_pts_edges = torch.stack((inv_idx, torch.arange(pts.shape[0], dtype=torch.long, device=pts.device)), dim=0)
    anchor_batch = scatter(pts_batch, anchor_pts_edges[0], reduce='min')

    # convert to 3D idx for sparse convs
    anchor_idx -= anchor_batch * max_grid_idx
    anchor_idx3d = torch.zeros((anchor_idx.shape[0], 3), dtype=torch.int, device=pts.device)
    anchor_idx3d[:, 2] = anchor_idx // (grid_size[0] * grid_size[1])
    anchor_idx3d[:, 1] = (anchor_idx - anchor_idx3d[:, 2] * (grid_size[0] * grid_size[1])) // (grid_size[0])
    anchor_idx3d[:, 0] = (anchor_idx - anchor_idx3d[:, 2] * (grid_size[0] * grid_size[1]))  % (grid_size[0])
    anchor_pts = anchor_idx3d * edge_len + bbox_min + edge_len / 2.

    # finally, make min of anchor idx3d [0, 0, 0] for every batch
    min_idx3d = scatter(anchor_idx3d, anchor_batch, dim=0, reduce='min')
    anchor_idx3d -= min_idx3d[anchor_batch]

    return anchor_pts, anchor_idx3d, anchor_batch, anchor_pts_edges


def build_img_pts(img_size=(240, 320), plane_size=(56, 56)):
    pts_x = np.linspace(0, img_size[1] - 1, plane_size[1], dtype=np.float32)
    pts_y = np.linspace(0, img_size[0] - 1, plane_size[0], dtype=np.float32)
    pts_xx, pts_yy = np.meshgrid(pts_x, pts_y)

    pts_xx = pts_xx.reshape(-1)
    pts_yy = pts_yy.reshape(-1)
    z = np.ones(pts_xx.shape[0], dtype=np.float32)

    pts = np.stack((pts_xx, pts_yy, z))
    return pts


def batched_build_img_pts_tensor(n_batch, img_size=(240, 320), plane_size=(60, 80)):
    pts = torch.from_numpy(build_img_pts(img_size, plane_size))
    pts = pts[None].repeat(n_batch, 1, 1)
    return pts


def batched_build_plane_sweep_volume_tensor(depth_start, depth_interval, n_planes, R, t, K, img_size=(240, 320),
                                            plane_size=(60, 80)):
    n_batch = R.shape[0]

    # build img points in numpy
    depth_end = depth_start + (n_planes - 1) * depth_interval
    pts_x = np.linspace(0, img_size[1]-1, plane_size[1], dtype=np.float32)
    pts_y = np.linspace(0, img_size[0]-1, plane_size[0], dtype=np.float32)
    z = np.linspace(depth_start, depth_end, n_planes, dtype=np.float32)
    pts_xx, pts_yy = np.meshgrid(pts_x, pts_y)
    pts = np.stack((pts_xx, pts_yy))
    pts = np.repeat(pts[:, None, :], z.shape[0], axis=1)
    pts = np.concatenate((pts, np.ones((1, z.shape[0], *plane_size))), axis=0)
    pts = pts * z[None, :, None, None]
    pts_tensor = torch.from_numpy(pts).float().view(3, -1).unsqueeze(0).repeat(n_batch, 1, 1).type_as(R)

    # perform batched matrix multiplies using pytorch tensors
    K_inv = torch.inverse(K)
    R_T = torch.transpose(R, 2, 1)
    pts_cam = torch.bmm(K_inv, pts_tensor)
    pts_world = torch.bmm(R_T, pts_cam - t[..., None])

    return pts_world


def get_propability_map(cv, depth_map, depth_start, depth_interval):
    """
    get probability map from cost volume
    COPIED FROM: https://github.com/callmeray/PointMVSNet
    """
    with torch.no_grad():
        batch_size, height, width = depth_map.shape
        depth = cv.size(1)

        # byx coordinates, batched & flattened
        b_coordinates = torch.arange(batch_size, dtype=torch.int64)
        y_coordinates = torch.arange(height, dtype=torch.int64)
        x_coordinates = torch.arange(width, dtype=torch.int64)
        b_coordinates = b_coordinates.view(batch_size, 1, 1).expand(batch_size, height, width)
        y_coordinates = y_coordinates.view(1, height, 1).expand(batch_size, height, width)
        x_coordinates = x_coordinates.view(1, 1, width).expand(batch_size, height, width)

        b_coordinates = b_coordinates.contiguous().view(-1).type(torch.long)
        y_coordinates = y_coordinates.contiguous().view(-1).type(torch.long)
        x_coordinates = x_coordinates.contiguous().view(-1).type(torch.long)

        # d coordinates (floored and ceiled), batched & flattened
        d_coordinates = ((depth_map - depth_start) / depth_interval).view(-1)
        d_coordinates = torch.detach(d_coordinates)
        d_coordinates_left0 = torch.clamp(d_coordinates.floor().long(), 0, depth - 1)
        d_coordinates_right0 = torch.clamp(d_coordinates.ceil().long(), 0, depth - 1)

    # get probability image by gathering
    prob_map_left0 = cv[b_coordinates, d_coordinates_left0, y_coordinates, x_coordinates]
    prob_map_right0 = cv[b_coordinates, d_coordinates_right0, y_coordinates, x_coordinates]

    prob_map = prob_map_left0 + prob_map_right0
    prob_map = prob_map.view(batch_size, height, width)

    return prob_map


def get_propability_map_from_flow(cv):
    """
    get probability map from cost volume
    COPIED FROM: https://github.com/callmeray/PointMVSNet
    """
    with torch.no_grad():
        batch_size, n_intervals, height, width = cv.shape

        # byx coordinates, batched & flattened
        b_coordinates = torch.arange(batch_size, dtype=torch.int64)
        y_coordinates = torch.arange(height, dtype=torch.int64)
        x_coordinates = torch.arange(width, dtype=torch.int64)
        b_coordinates = b_coordinates.view(batch_size, 1, 1).expand(batch_size, height, width)
        y_coordinates = y_coordinates.view(1, height, 1).expand(batch_size, height, width)
        x_coordinates = x_coordinates.view(1, 1, width).expand(batch_size, height, width)

        b_coordinates = b_coordinates.contiguous().view(-1).type(torch.long)
        y_coordinates = y_coordinates.contiguous().view(-1).type(torch.long)
        x_coordinates = x_coordinates.contiguous().view(-1).type(torch.long)

        # d coordinates (floored and ceiled), batched & flattened
        interval = torch.arange(n_intervals).type_as(cv)
        d_coordinates = torch.sum(cv * interval[None, :, None, None], dim=1).view(-1)
        d_coordinates = torch.detach(d_coordinates)
        d_coordinates_left0 = torch.clamp(d_coordinates.floor().long(), 0, n_intervals - 1)
        d_coordinates_right0 = torch.clamp(d_coordinates.ceil().long(), 0, n_intervals - 1)

    # get probability image by gathering
    prob_map_left0 = cv[b_coordinates, d_coordinates_left0, y_coordinates, x_coordinates]
    prob_map_right0 = cv[b_coordinates, d_coordinates_right0, y_coordinates, x_coordinates]

    prob_map = prob_map_left0 + prob_map_right0
    prob_map = prob_map.view(batch_size, height, width)

    return prob_map
