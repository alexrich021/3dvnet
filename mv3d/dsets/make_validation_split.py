from mv3d.dsets.scenelists import get_scenes_scannet
from mv3d import config
import numpy as np
import os

N_SCENES = 100
NAME = 'val_split1.txt'

scene_split_save_dir = os.path.join(os.path.dirname(__file__), 'scannet_splits')
if not os.path.exists(scene_split_save_dir):
    os.makedirs(scene_split_save_dir)
scene_split_filename = os.path.join(scene_split_save_dir, NAME)
if os.path.exists(scene_split_filename):
    raise FileExistsError('Please manually delete existing split to avoid unwanted overwrite')

all_val_scenes = get_scenes_scannet(config.SCANNET_DIR, 'val')
n_val_scense = len(all_val_scenes)
random_idx_shuffle = np.random.permutation(np.arange(n_val_scense))
first_n_idx = random_idx_shuffle[:N_SCENES]
scenes = [os.path.basename(s)+'\n' for i, s in enumerate(all_val_scenes) if i in first_n_idx]

with open(scene_split_filename, 'w') as f:
    f.writelines(scenes)
