import torch
from mv3d.dsets import dataset
from mv3d.dsets import scenelists
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from mv3d import config
from mv3d.baselines.fastmvsnet.lightningmodel import FastMVSNetPLModel
from mv3d.dsets import frameselector

fastmvsnet = FastMVSNetPLModel(config.FINETUNE_LR, config.FINETUNE_LR_STEP, config.FINETUNE_LR_GAMMA)
fastmvsnet.load_pretrained_fastmvsnet()

kwargs = dict()
if torch.cuda.is_available():
    kwargs['gpus'] = config.GPUS
if config.USE_WANDB:
    wandb_logger = WandbLogger(project='3dvnet-logs')
    kwargs['logger'] = wandb_logger

train_scenes = scenelists.get_scenes_scannet(config.SCANNET_DIR, 'train')
val_scenes = scenelists.get_scenes_scannet(config.SCANNET_DIR, 'val')

train_selector = frameselector.RangePoseDistSelector(0.125, 0.325, config.IMG_INTERVAL)
val_selector = frameselector.BestPoseDistSelector(0.225, config.IMG_INTERVAL)

dset = dataset.Dataset(
    train_scenes, train_selector, config.N_REF_IMGS, (448, 640), (448, 640), True,
    mean_rgb=[0., 0., 0.], std_rgb=[1., 1., 1.], scale_rgb=1.)
val_dset = dataset.Dataset(
    val_scenes, val_selector, config.N_REF_IMGS, (448, 640), (448, 640), False,
    mean_rgb=[0., 0., 0.], std_rgb=[1., 1., 1.], scale_rgb=1.)

loader = dataset.get_dataloader(dset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = dataset.get_dataloader(val_dset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8)

trainer = pl.Trainer(**kwargs)
trainer.fit(fastmvsnet, loader, val_loader)
