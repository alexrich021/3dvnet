import os


def get_scenes_scannet(scannet_dir, split='train'):
    scenes_dir = os.path.join(scannet_dir, 'scans_test' if split == 'test' else 'scans')
    if split in ['train', 'val', 'test']:
        split_txt = os.path.join(scannet_dir, 'scannetv2_{}.txt'.format(split))
    else:
        split_txt = os.path.join(os.path.dirname(__file__), 'scannet_splits', '{}.txt'.format(split))
    with open(split_txt, 'r') as fp:
        scenes_list = [os.path.join(scenes_dir, f.strip()) for f in fp.readlines()]
    return scenes_list


def get_scenes_icl_nuim(icl_nuim_dir):
    return [
        os.path.join(icl_nuim_dir, 'living_room_traj1_frei_png'),  # living room 1
        os.path.join(icl_nuim_dir, 'living_room_traj2_frei_png'),  # living room 2
        os.path.join(icl_nuim_dir, 'traj1_frei_png'),              # office 1
        os.path.join(icl_nuim_dir, 'traj2_frei_png'),              # office 2
    ]


def get_scenes_tum_rgbd(tum_rgbd_dir):
    return [
        os.path.join(tum_rgbd_dir, 'rgbd_dataset_freiburg1_desk'),
        os.path.join(tum_rgbd_dir, 'rgbd_dataset_freiburg1_plant'),
        os.path.join(tum_rgbd_dir, 'rgbd_dataset_freiburg1_room'),
        os.path.join(tum_rgbd_dir, 'rgbd_dataset_freiburg1_teddy'),
        os.path.join(tum_rgbd_dir, 'rgbd_dataset_freiburg2_desk'),
        os.path.join(tum_rgbd_dir, 'rgbd_dataset_freiburg2_dishes'),
        os.path.join(tum_rgbd_dir, 'rgbd_dataset_freiburg3_cabinet'),
        os.path.join(tum_rgbd_dir, 'rgbd_dataset_freiburg3_long_office_household'),
        os.path.join(tum_rgbd_dir, 'rgbd_dataset_freiburg3_structure_notexture_far'),
        os.path.join(tum_rgbd_dir, 'rgbd_dataset_freiburg3_structure_texture_far'),
    ]
