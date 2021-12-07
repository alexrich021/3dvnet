import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from mv3d import config
from mv3d.dsets import dataset
from mv3d.dsets.scenelists import get_scenes_scannet
from mv3d.lightningmodel import PL3DVNet
from mv3d.dsets import frameselector

pl3dvnet = PL3DVNet(config.DEPTH_TRAIN, config.DEPTH_TEST, config.GRID_EDGE_LEN, config.IMG_FEAT_DIM, config.IMG_SIZE,
                    lr=1e-3, lr_step=100, lr_gamma=0.1, finetune=False)

kwargs = dict()
if torch.cuda.is_available():
    kwargs['gpus'] = config.GPUS
if config.USE_WANDB:
    wandb_logger = WandbLogger(project='3dvnet-logs')
    kwargs['logger'] = wandb_logger

train_selector = frameselector.RangePoseDistSelector(0.125, 0.325, config.IMG_INTERVAL)
val_selector = frameselector.BestPoseDistSelector(0.225, config.IMG_INTERVAL)

train_scenes = get_scenes_scannet(config.SCANNET_DIR, 'train')
val_scenes = get_scenes_scannet(config.SCANNET_DIR, 'val')
dset = dataset.Dataset(train_scenes, train_selector, config.N_REF_IMGS, config.IMG_SIZE, config.IMG_SIZE, True,
                       crop=False)
val_dset = dataset.Dataset(val_scenes, val_selector, config.N_REF_IMGS, config.IMG_SIZE, config.IMG_SIZE, False,
                           crop=False)

loader = dataset.get_dataloader(dset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = dataset.get_dataloader(val_dset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8)

trainer = pl.Trainer(**kwargs)
trainer.fit(pl3dvnet, loader, val_loader)
