import numpy as np
import torch.utils.data as data
import cv2
from path import Path
import random
import imageio
from mv3d.dsets.scannetdset import get_scenes
import json
import tqdm


def Rt2pose(Rt):
    R = Rt[:3,:3]
    t = Rt[:3, 3]
    R = R.transpose()
    t = -np.dot(R, t)
    return np.vstack((np.hstack((R,t.reshape(3,1))), np.array([0,0,0,1])))


def pose_distance(p1, p2):
    rel_pose = np.dot(np.linalg.inv(p1), p2)
    R = rel_pose[:3, :3]
    t = rel_pose[:3, 3]

    return round(np.sqrt(np.linalg.norm(t) ** 2 + 2 * (1 - min(3.0, np.matrix.trace(R)) / 3)), 4)


def genDistM(poses):
    n = len(poses)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = pose_distance(poses[i], poses[j])
    return D


def crawl_folders(folders_list, sequence_length):
    sequence_set = []
    seq_len = sequence_length
    for folder in tqdm.tqdm(folders_list):
        info = json.load(open(folder / 'info.json', 'r'))
        # print(list(info.keys()))
        # print(list(info['frames'][0].keys()))
        intrinsics = np.asarray(info['intrinsics'])
        imgs = [Path(f['filename_color']) for f in info['frames']]
        gt = [Path(f['filename_depth']) for f in info['frames']]
        poses = [np.asarray(f['pose']) for f in info['frames']]
        # imgs = sorted(folder.files('*.png'))
        #
        # poses = []
        # with open(folder / 'pose.txt') as f:
        #     for l in f.readlines():
        #         l = l.strip('\n')
        #         poses.append(np.array(l.split(' ')).astype(np.float32).reshape(4, 4))
        #
        if len(imgs) < sequence_length:  #to load test set
            seq_len = 2

        for i in range(0, len(imgs) - seq_len + 1):
            sample = {'intrinsics': intrinsics,
                      'imgs': [imgs[i]],
                      'gts': [gt[i]],
                      'poses':[Rt2pose(poses[i])]
            }

            for j in range(1, seq_len):
                   sample['imgs'].append(imgs[i + j])
                   src_pose = Rt2pose(poses[i + j])
                   sample['poses'].append(src_pose)
                   sample['gts'].append(gt[i + j])
            sequence_set.append(sample)

    random.shuffle(sequence_set)
    return sequence_set


def load_as_float(path):
    return cv2.imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    def __init__(self, root, seed=None, split='train', sequence_length=3,transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.scenes = [Path(s) for s in get_scenes(root, split)][:5]
        self.samples = crawl_folders(self.scenes, sequence_length)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [load_as_float(ref_img) for ref_img in sample['imgs']]

        poses = sample['poses']

        left2rights = []
        for i in range(0, len(poses)-1):
            left2rights.append(np.dot(np.linalg.inv(poses[i+1]), poses[i]))

        left2rights.append(np.dot(np.linalg.inv(poses[-2]), poses[-1]))  #last frame use former frame as neighbour

        gts = []
        for gt_path in sample['gts']:
            print(gt_path)
            gt = imageio.imread(gt_path)
            valid_mask = gt != 0.0
            gt = np.where(valid_mask, 1.0 / gt, 0.0)  # get disp map
            gts.append(gt)

        camera_k = sample['intrinsics']
        original_width = imgs[0].shape[1]
        original_height = imgs[0].shape[0]
        factor_x = 320.0 / original_width
        factor_y = 256.0 / original_height

        imgs = [cv2.resize(img, (320, 256)) for img in imgs]

        imgs = [((img - 81.0) / 35.0) for img in imgs]

        camera_k[0, :] *= factor_x
        camera_k[1, :] *= factor_y

        if self.transform is not None:
            imgs, K = self.transform(imgs, np.copy(camera_k))
        else:
            K = np.copy(camera_k)

        left_in_right_Ts = [left2right[0:3, 3] for left2right in left2rights]
        left_in_right_Rs = [left2right[0:3, 0:3] for left2right in left2rights]

        pixel_coordinate = np.indices([320, 256]).astype(np.float32)
        pixel_coordinate = np.concatenate((pixel_coordinate, np.ones([1, 320, 256])), axis=0)
        pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])

        KRK_is = [K.dot(left_in_right_R.dot(np.linalg.inv(K))) for left_in_right_R in left_in_right_Rs]
        KRKiUVs = [KRK_i.dot(pixel_coordinate) for KRK_i in KRK_is]
        KTs = [K.dot(left_in_right_T) for left_in_right_T in left_in_right_Ts]
        KTs = [np.expand_dims(KT, -1) for KT in KTs]

        KTs = [KT.astype(np.float32) for KT in KTs]

        KRKiUVs = [KRKiUV.astype(np.float32) for KRKiUV in KRKiUVs]

        D = genDistM(poses)
        return imgs, KRKiUVs, KTs, gts, D

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    import open3d as o3d
    import matplotlib.pyplot as plt
    dset = SequenceFolder('/home/alex/Desktop/scannet', split='test')
    imgs, KRKiUVs, KTs, gts, D = dset[0]
    print(len(imgs), imgs[0].shape)
    print(len(gts), gts[0].shape)
    print(imgs[0].max(), imgs[0].min())
    exit()
    plt.subplot(2, 3, 1)
    plt.imshow(imgs[0])
    plt.subplot(2, 3, 2)
    plt.imshow(imgs[1])
    plt.subplot(2, 3, 3)
    plt.imshow(imgs[2])
    plt.subplot(2, 3, 4)
    plt.imshow(gts[0])
    plt.subplot(2, 3, 5)
    plt.imshow(gts[1])
    plt.subplot(2, 3, 6)
    plt.imshow(gts[2])
    plt.show()
