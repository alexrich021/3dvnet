# Standard options
USE_WANDB = True       # if false, does not use WANDB logging
GPUS = 1               # number of GPUs to use. All experiments in paper were run on a single 3090 GPU. The code is not
                       # currently set up for more than 1 GPU, but the helper classes in mv3d/lightningplugins.py
                       # should help with setup should you want to use more than 1

# dataset dir locations
SCANNET_DIR = '/home/alex/Desktop/scannet'  # dir containing scans/ and scans_test/ dirs produced by preproc script
ICL_NUIM_DIR = '/home/alex/Desktop/icl-nuim'
TUM_RGBD_DIR = '/home/alex/Desktop/tum-rgbd'

# Training settings
BATCH_SIZE = 2
LAMBDA = 0.5

# Baseline finetuning settings
FINETUNE_LR = 1e-4
FINETUNE_LR_STEP = 50
FINETUNE_LR_GAMMA = 0.5

# 3D reconstruction options
GRID_EDGE_LEN = 0.08  # voxel resolution for scene-modeling step

# 2D depth prediction options
IMG_SIZE = (256, 320)
DEPTH_TEST = {
    'depth_start': 0.5,
    'depth_interval': 0.05,
    'n_intervals': 96,
    'size': (56, 56)
}
DEPTH_TRAIN = {
    'depth_start': 0.5,
    'depth_interval': 0.05,
    'n_intervals': 96,
    'size': (56, 56)
}
N_REF_IMGS = 7
IMG_INTERVAL = 20

# Model dimension settings
IMG_FEAT_DIM = 32

# specifies path to 3dvnet model prior to finetuning of pre-trained CNN
# (note this example path points to a model that has already been fine-tuned)
PATH = '/home/alex/Desktop/3dvnet_code/mv3d/3dvnet_weights/epoch=100-step=60700.ckpt'
