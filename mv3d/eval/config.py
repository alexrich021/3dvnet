DATASET_TYPE = 'scannet'

# specifiy where evaluation results should be saved
if DATASET_TYPE == 'icl-nuim':
    SAVE_DIR = '/media/ssd0/icl-nuim_results/'
elif DATASET_TYPE == 'tum-rgbd':
    SAVE_DIR = '/media/ssd0/tum-rgbd_results/'
elif DATASET_TYPE == 'scannet_val':
    SAVE_DIR = '/media/ssd0/val_results/'
elif DATASET_TYPE == 'scannet':
    SAVE_DIR = '/home/alex/Desktop/tmp/results/'
else:
    SAVE_DIR = None
    raise Exception

# frame selection options
PDIST = 0.1                 # pose distance for adding keyframe
N_SRC_ON_EITHER_SIDE = 2    # number of source views to use on either side of reference view in keyframe sequence

# Reconstruction options
RUN_PCFUSION = True
RUN_TSDF_FUSION = False
MASK_USING_GT_MESH = True   # masks missing regions in observed space during reconstruction

# Depth-based 3D options
Z_THRESH = 0.01             # pc fusion multi-view consistency threshold
N_CONSISTENT_THRESH = 3     # pc-fusion threshold on number of consistent views
VOXEL_DOWNSAMPLE = 0.02     # for point cloud voxelization of GT mesh
DIST_THRESH = 0.05          # threshold for calculating prec/recall/fscore

# Volumetric-based 3D options (copied from Atlas evaluation)
IMG_BATCH = 100
VOX_RES = 0.04
VOL_PRCNT = .995
VOL_MARGIN = 1.5
TRUNC_RATIO = 3

FUSIBILE_EXE_PATH = '/home/alex/Desktop/cloned_paper_code/mvsnet-fusibile/build_depth/fusibile'
