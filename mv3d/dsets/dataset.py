from torch_geometric import data
import torch
import torch.nn.functional as F
from torchvision import transforms as tv_transforms
import os
import numpy as np
from mv3d import utils
from mv3d.dsets.batch import Batch
import json
import cv2
import random
from kornia import adjust_brightness, adjust_gamma, adjust_contrast


img_transforms = tv_transforms.Compose([
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PreprocessImage:
    def __init__(self, K, old_width, old_height, new_width, new_height, distortion_crop=0, perform_crop=True):
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.new_width = new_width
        self.new_height = new_height
        self.perform_crop = perform_crop

        original_height = np.copy(old_height)
        original_width = np.copy(old_width)

        if self.perform_crop:
            old_height -= 2 * distortion_crop
            old_width -= 2 * distortion_crop

            old_aspect_ratio = float(old_width) / float(old_height)
            new_aspect_ratio = float(new_width) / float(new_height)

            if old_aspect_ratio > new_aspect_ratio:
                # we should crop horizontally to decrease image width
                target_width = old_height * new_aspect_ratio
                self.crop_x = int(np.floor((old_width - target_width) / 2.0)) + distortion_crop
                self.crop_y = distortion_crop
            else:
                # we should crop vertically to decrease image height
                target_height = old_width / new_aspect_ratio
                self.crop_x = distortion_crop
                self.crop_y = int(np.floor((old_height - target_height) / 2.0)) + distortion_crop

            self.cx -= self.crop_x
            self.cy -= self.crop_y
            intermediate_height = original_height - 2 * self.crop_y
            intermediate_width = original_width - 2 * self.crop_x

            factor_x = float(new_width) / float(intermediate_width)
            factor_y = float(new_height) / float(intermediate_height)

            self.fx *= factor_x
            self.fy *= factor_y
            self.cx *= factor_x
            self.cy *= factor_y
        else:
            self.crop_x = 0
            self.crop_y = 0
            factor_x = float(new_width) / float(original_width)
            factor_y = float(new_height) / float(original_height)

            self.fx *= factor_x
            self.fy *= factor_y
            self.cx *= factor_x
            self.cy *= factor_y

    def apply_depth(self, depth):
        raw_height, raw_width = depth.shape
        cropped_depth = depth[self.crop_y:raw_height - self.crop_y, self.crop_x:raw_width - self.crop_x]
        resized_cropped_depth = cv2.resize(cropped_depth, (self.new_width, self.new_height), interpolation=cv2.INTER_NEAREST)
        return resized_cropped_depth

    def apply_rgb(self, image, scale_rgb, mean_rgb, std_rgb, normalize_colors=True):
        raw_height, raw_width, _ = image.shape
        cropped_image = image[self.crop_y:raw_height - self.crop_y, self.crop_x:raw_width - self.crop_x, :]
        cropped_image = cv2.resize(cropped_image, (self.new_width, self.new_height), interpolation=cv2.INTER_LINEAR)

        if normalize_colors:
            cropped_image = cropped_image / scale_rgb
            cropped_image[:, :, 0] = (cropped_image[:, :, 0] - mean_rgb[0]) / std_rgb[0]
            cropped_image[:, :, 1] = (cropped_image[:, :, 1] - mean_rgb[1]) / std_rgb[1]
            cropped_image[:, :, 2] = (cropped_image[:, :, 2] - mean_rgb[2]) / std_rgb[2]
        return cropped_image

    def get_updated_intrinsics(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]])


class Dataset(data.Dataset):
    def __init__(self, scene_dirs, frame_selector, n_ref_imgs=None, depth_img_size=(56, 56), img_size=(256, 320),
                 augment=False, scale_rgb=255., mean_rgb=[0.485, 0.456, 0.406], std_rgb=[0.229, 0.224, 0.225],
                 n_src_on_either_side=1, crop=False):
        super(Dataset, self).__init__()
        self.scene_dirs = scene_dirs
        self.n_ref_imgs = n_ref_imgs
        self.depth_img_size = depth_img_size
        self.img_size = img_size
        self.augment = augment
        self.n_src_on_either_side = n_src_on_either_side
        self.frame_selector = frame_selector
        self.crop = crop

        self.scale_rgb = scale_rgb
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb

    def len(self):
        return len(self.scene_dirs)

    def get(self, idx, return_verbose=False, seed_idx=None):
        # getting necessary directory information
        scene_dir = self.scene_dirs[idx]

        scene_info = json.load(open(os.path.join(scene_dir, 'info.json'), 'r'))
        all_poses = np.stack([np.asarray(frame['pose']) for frame in scene_info['frames']], axis=0)
        K = np.asarray(scene_info['intrinsics'])

        n_imgs = self.n_ref_imgs+2*self.n_src_on_either_side if self.n_ref_imgs is not None else 100_000
        img_idx = self.frame_selector.select_frames(all_poses, n_imgs, seed_idx)
        ref_idx = img_idx[self.n_src_on_either_side:-self.n_src_on_either_side]
        n_ref_imgs = ref_idx.shape[0]

        n_imgs_per_ref = 2*self.n_src_on_either_side+1
        ref_src_edges = torch.empty((2, n_ref_imgs * n_imgs_per_ref), dtype=torch.long)
        for i in range(n_ref_imgs):
            ref_src_edges[0, i * n_imgs_per_ref: (i + 1) * n_imgs_per_ref] = torch.ones(n_imgs_per_ref) * (i + self.n_src_on_either_side)
            ref_src_edges[1, i * n_imgs_per_ref: (i + 1) * n_imgs_per_ref] = torch.arange(i, i + n_imgs_per_ref)

        raw_images = []
        raw_depths = []
        for i in img_idx:
            frame_info = scene_info['frames'][i]
            color = cv2.imread(frame_info['filename_color'])
            depth = cv2.imread(frame_info['filename_depth'], cv2.IMREAD_ANYDEPTH)
            raw_images.append(color)
            raw_depths.append(depth)

        preprocessor = PreprocessImage(K=K,
                                       old_width=raw_images[0].shape[1],
                                       old_height=raw_images[0].shape[0],
                                       new_width=self.img_size[1],
                                       new_height=self.img_size[0],
                                       distortion_crop=0,
                                       perform_crop=self.crop)

        rgb_sum = 0
        intermediate_depths = []
        intermediate_images = []
        for i in range(len(raw_images)):
            depth = (raw_depths[i]).astype(np.float32) / 1000.0
            depth_nan = depth == np.nan
            depth_inf = depth == np.inf
            depth_outofrange = depth > 65.  # 7scenes stores invalid depths as 65_535mm
            depth_invalid = depth_inf | depth_nan | depth_outofrange
            depth[depth_invalid] = 0
            depth = preprocessor.apply_depth(depth)
            intermediate_depths.append(depth)

            image = raw_images[i]
            image = preprocessor.apply_rgb(image=image,
                                           scale_rgb=1.0,
                                           mean_rgb=[0.0, 0.0, 0.0],
                                           std_rgb=[1.0, 1.0, 1.0],
                                           normalize_colors=False)
            rgb_sum += np.sum(image)
            intermediate_images.append(image)
        rgb_average = rgb_sum / (len(raw_images) * self.img_size[0] * self.img_size[1] * 3)

        # COLOR AUGMENTATION
        color_transforms = []
        brightness = random.uniform(-0.03, 0.03)
        contrast = random.uniform(0.8, 1.2)
        gamma = random.uniform(0.8, 1.2)
        color_transforms.append((adjust_gamma, gamma))
        color_transforms.append((adjust_contrast, contrast))
        color_transforms.append((adjust_brightness, brightness))
        random.shuffle(color_transforms)

        K = preprocessor.get_updated_intrinsics()
        depth_images = []
        images = []
        for i in range(len(raw_images)):
            image = intermediate_images[i]
            depth = intermediate_depths[i]
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image.astype(np.float32))
            image = image / 255.0
            if self.augment and (55.0 < rgb_average < 200.0):
                for (color_transform_function, color_transform_value) in color_transforms:
                    image = color_transform_function(image, color_transform_value)

            image = (image * 255.0) / self.scale_rgb
            image[0, :, :] = (image[0, :, :] - self.mean_rgb[0]) / self.std_rgb[0]
            image[1, :, :] = (image[1, :, :] - self.mean_rgb[1]) / self.std_rgb[1]
            image[2, :, :] = (image[2, :, :] - self.mean_rgb[2]) / self.std_rgb[2]

            images.append(image)

            depth = torch.from_numpy(depth.astype(np.float32))
            depth_images.append(depth)

        depth_images = torch.stack(depth_images, dim=0)
        images = torch.stack(images, dim=0)
        rotmats = torch.from_numpy(all_poses[img_idx, :3, :3]).float().transpose(2, 1)
        tvecs = -torch.bmm(rotmats, torch.from_numpy(all_poses[img_idx, :3, 3, None]).float())[..., 0]
        K = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).expand(images.shape[0], 3, 3)
        if self.n_src_on_either_side > 0:
            depth_images = depth_images[self.n_src_on_either_side: -self.n_src_on_either_side]
        depth_images = F.interpolate(depth_images.unsqueeze(1), self.depth_img_size, mode='nearest').squeeze(1)

        # dsets augmentation
        # random rotation about gravitational axis
        R_aug = torch.from_numpy(utils.random_gravitational_rotation_scannet()) if self.augment \
            else torch.eye(3, dtype=torch.float32)

        rotmats = rotmats @ R_aug.T

        # scale aug
        S_aug = random.uniform(0.9, 1.1) if self.augment else 1.
        depth_images = depth_images * S_aug
        tvecs = tvecs * S_aug

        batch = Batch(images, rotmats, tvecs, K, depth_images, ref_src_edges)
        if return_verbose:
            return batch, R_aug, S_aug, img_idx
        else:
            return batch


def get_dataloader(dset, batch_size, **kwargs):
    return data.DataLoader(dset, follow_batch=['images', 'query_pts'],
                           batch_size=batch_size, **kwargs)


def get_datalistloader(dset, batch_size, **kwargs):
    return data.DataListLoader(dset, batch_size=batch_size, **kwargs)


if __name__ == '__main__':
    """
    Open3D visualization for debugging - see key callbacks at bottom of script 
    """
    import open3d as o3d
    import vis_utils
    import glob
    from mv3d.dsets import scenelists
    from mv3d.dsets import frameselector

    # scene_dirs = scenelists.get_scenes_scannet(config.SCANNET_DIR, 'test')
    scene_dirs = ['/home/alex/Desktop/scene0000_00_dst', ]
    selector = frameselector.BestPoseDistSelector(0.225, 20)
    dset = Dataset(scene_dirs, selector, 15, (256, 320), (256, 320), False, 255., [0., 0., 0.], [1., 1., 1.], crop=False)
    s_idx = random.randint(0, len(dset) - 1)
    b, R, S, _ = dset.get(s_idx, True)
    R = R.numpy()
    print('{}: {}'.format(s_idx, os.path.basename(dset.scene_dirs[s_idx])))
    print('S: {}'.format(S))
    print(torch.max(b.depth_images), torch.min(b.depth_images))
    ref_idx = torch.unique(b.ref_src_edges[0])
    pts_world = utils.batched_build_plane_sweep_volume_tensor(0.5, 0.4, 12, b.rotmats[ref_idx], b.tvecs[ref_idx],
                                                              b.K[ref_idx], (256, 320), plane_size=(32, 32))
    pts_world = pts_world.transpose(2, 1)  # b x n_pts x 3

    # for i in range(b.images.shape[0]):
    #     cv2.imshow(str(i), b.images[i].permute(1, 2, 0).numpy())
    # cv2.waitKey(0)

    def get_frustrum_pcd(idx):
        pts = pts_world[idx]
        pcd = vis_utils.numpy_arr_to_o3d_pc(pts)
        return pcd

    idx = 0
    pts_pcd = get_frustrum_pcd(idx)

    def iter_pts_pcd(vis, iter=+1):
        global idx
        global pts_pcd
        vis_utils.remove_pcd(vis, pts_pcd)
        idx = (idx + iter) % pts_world.shape[0]
        print(idx)
        pts_pcd = get_frustrum_pcd(idx)
        vis_utils.add_pcd(vis, pts_pcd)

    scene_dir = dset.scene_dirs[s_idx]
    try:
        scene_info = json.load(open(os.path.join(scene_dirs[s_idx], 'info.json'), 'r'))
        mesh = o3d.io.read_triangle_mesh(scene_info['gt_mesh'])
        verts = np.asarray(mesh.vertices)
        verts = S * (verts @ R.T)
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.compute_vertex_normals()
    except:
        print('Mesh not found')
        mesh = o3d.geometry.TriangleMesh()

    cam_pts = torch.bmm(-b.rotmats.transpose(2, 1), b.tvecs[..., None])[..., 0]
    cam_spheres = [vis_utils.create_sphere(pt) for pt in cam_pts[1: -1]] + \
                  [vis_utils.create_sphere(pt, color=(0.1, 0.9, 0.1)) for pt in cam_pts[[0, -1]]]

    n_imgs = b.depth_images.shape[0]
    K_inv = torch.inverse(b.K[ref_idx])
    R_T = b.rotmats[ref_idx].transpose(2, 1)
    dpts = utils.batched_build_img_pts_tensor(n_imgs, (256, 320), b.depth_images.shape[1:])
    dpts = dpts * b.depth_images.view(n_imgs, 1, -1)
    dpts = torch.bmm(R_T, torch.bmm(K_inv, dpts) - b.tvecs[ref_idx].unsqueeze(-1))
    dpts = dpts.transpose(2, 1).reshape(-1, 3).numpy()
    d_pcd = vis_utils.numpy_arr_to_o3d_pc(dpts)

    callbacks = dict()
    callbacks[ord('S')] = lambda x: vis_utils.remove_pcd(x, mesh)
    callbacks[ord('Z')] = lambda x: vis_utils.add_pcd(x, mesh)
    callbacks[ord('D')] = lambda x: vis_utils.remove_pcd(x, pts_pcd)
    callbacks[ord('X')] = lambda x: vis_utils.add_pcd(x, pts_pcd)
    callbacks[ord('F')] = lambda x: vis_utils.remove_pcd(x, d_pcd)
    callbacks[ord('C')] = lambda x: vis_utils.add_pcd(x, d_pcd)
    callbacks[ord('G')] = lambda x: vis_utils.remove_pcds(x, cam_spheres)
    callbacks[ord('V')] = lambda x: vis_utils.add_pcds(x, cam_spheres)
    callbacks[ord('H')] = lambda x: iter_pts_pcd(x, +1)
    callbacks[ord('B')] = lambda x: iter_pts_pcd(x, -1)

    o3d.visualization.draw_geometries_with_key_callbacks([d_pcd],
                                                         callbacks)
    cv2.destroyAllWindows()
