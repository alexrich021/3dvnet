from mv3d.lightningmodel import PL3DVNet
from mv3d.dsets.batch import Batch
from mv3d import utils
from mv3d.eval.main import main
import torch
import torch.nn.functional as F
import tqdm

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# batch options (change these to scale eval script to your GPU)
INIT_DEPTH_BATCH = 18
OFFSET_BATCH = 16
UPSAMPLE_BATCH = 100

# depth prediction and refinement options
DEPTH_CONFIG = {
    'depth_start': 0.5,
    'depth_interval': 0.05,
    'n_intervals': 96,
    'size': (56, 56)
}
OFFSETS_LIST = [[0.05, 0.05, 0.025], [0.05, 0.05, 0.025]]


def process_scene(batch, scene, dset, net):
    with torch.no_grad():

        print('Making initial depth predictions')
        ref_idx, gather_idx = torch.unique(batch.ref_src_edges[0], return_inverse=True)
        n_ref_imgs = len(ref_idx)
        n_imgs = batch.images.shape[0]
        n_mvs_batches = (n_ref_imgs - 1) // INIT_DEPTH_BATCH + 1

        all_depth = torch.empty((n_ref_imgs, *net.hparams.depth_test['size']), dtype=torch.float32, device=DEVICE)
        all_feats_half = torch.empty((n_imgs, net.hparams.feat_dim, net.hparams.img_size[0] // 2,
                                      net.hparams.img_size[1] // 2), dtype=torch.float32, device=DEVICE)
        all_feats_quarter = torch.empty((n_imgs, net.hparams.feat_dim, net.hparams.img_size[0] // 4,
                                        net.hparams.img_size[1] // 4), dtype=torch.float32, device=DEVICE)

        for i in tqdm.tqdm(range(n_mvs_batches)):
            ref_idx_start = i * INIT_DEPTH_BATCH + dset.n_src_on_either_side
            ref_idx_end = min((i + 1) * INIT_DEPTH_BATCH, n_ref_imgs) + dset.n_src_on_either_side
            idx_start = ref_idx_start - dset.n_src_on_either_side
            idx_end = ref_idx_end + dset.n_src_on_either_side

            images = batch.images[idx_start: idx_end]
            rotmats = batch.rotmats[idx_start: idx_end]
            tvecs = batch.tvecs[idx_start: idx_end]
            K = batch.K[idx_start: idx_end]
            ref_src_edges = utils.slice_edges(batch.ref_src_edges, ref_idx_start, ref_idx_end, 0)
            ref_src_edges -= idx_start

            batch_sliced = Batch(images, rotmats, tvecs, K, None, ref_src_edges)
            batch_sliced.__setattr__('images_batch', torch.zeros(images.shape[0], dtype=torch.long))
            batch_sliced.to(DEVICE)

            pred, depth_batch, feats_half, feats_quarter, feats_eighth, _ = \
                net.make_initial_depth_predictions(batch_sliced, DEPTH_CONFIG)

            all_depth[i * INIT_DEPTH_BATCH: (i + 1) * INIT_DEPTH_BATCH] = pred.detach()
            all_feats_half[idx_start: idx_end] = feats_half.detach()
            all_feats_quarter[idx_start: idx_end] = feats_quarter.detach()

        print('Refining depth pts')
        rotmats = batch.rotmats.to(DEVICE)
        tvecs = batch.tvecs.to(DEVICE)
        K = batch.K.to(DEVICE)
        ref_src_edges = batch.ref_src_edges.to(DEVICE)
        depth_batch = torch.zeros(all_depth.shape[0], dtype=torch.long, device=DEVICE)

        n_offset_batches = (n_ref_imgs - 1) // OFFSET_BATCH + 1
        for i, offsets in enumerate(OFFSETS_LIST):
            print('Iter {} / {}'.format(i + 1, len(OFFSETS_LIST)))
            xs, pts = net.model_scene(all_depth, depth_batch, all_feats_quarter, rotmats, tvecs, K, ref_src_edges,
                                      return_pts=True)

            for offset in offsets:
                for b in tqdm.tqdm(range(n_offset_batches)):
                    ref_idx_start = b * OFFSET_BATCH + dset.n_src_on_either_side
                    ref_idx_end = min((b + 1) * OFFSET_BATCH, n_ref_imgs) + dset.n_src_on_either_side
                    idx_start = ref_idx_start - dset.n_src_on_either_side
                    idx_end = ref_idx_end + dset.n_src_on_either_side

                    edges = utils.slice_edges(ref_src_edges, ref_idx_start, ref_idx_end, 0)
                    edges -= idx_start

                    offset_pred = net.run_pointflow(
                        xs,
                        all_depth[ref_idx_start - dset.n_src_on_either_side: ref_idx_end - dset.n_src_on_either_side],
                        depth_batch[ref_idx_start - dset.n_src_on_either_side: ref_idx_end - dset.n_src_on_either_side],
                        all_feats_quarter[idx_start: idx_end],
                        rotmats[idx_start: idx_end],
                        tvecs[idx_start: idx_end],
                        K[idx_start: idx_end],
                        edges,
                        offset,
                        3)
                    all_depth[ref_idx_start - dset.n_src_on_either_side: ref_idx_end - dset.n_src_on_either_side] += offset_pred

        print('Upsampling depth predictions')
        # starting size --> 1/4
        all_depth = F.interpolate(all_depth.unsqueeze(1), all_feats_quarter.shape[-2:], mode='nearest').squeeze(1)
        n_batches = (n_ref_imgs - 1) // UPSAMPLE_BATCH + 1
        for b in range(n_batches):
            idx_start = b * UPSAMPLE_BATCH
            idx_end = (b+1) * UPSAMPLE_BATCH
            all_depth[idx_start: idx_end] = net.refine_quarter(all_feats_quarter[ref_idx][idx_start: idx_end],
                                                               all_depth[idx_start: idx_end].unsqueeze(1)).squeeze(1)
        # 1/4 --> 1/2
        all_depth = F.interpolate(all_depth.unsqueeze(1), all_feats_half.shape[-2:], mode='nearest').squeeze(1)
        n_batches = (n_ref_imgs - 1) // UPSAMPLE_BATCH + 1
        for b in range(n_batches):
            idx_start = b * UPSAMPLE_BATCH
            idx_end = (b+1) * UPSAMPLE_BATCH
            all_depth[idx_start: idx_end] = net.refine_half(all_feats_half[ref_idx][idx_start: idx_end],
                                                            all_depth[idx_start: idx_end].unsqueeze(1)).squeeze(1)
        # 1/2 --> image size
        all_depth = F.interpolate(all_depth.unsqueeze(1), batch.images.shape[-2:], mode='nearest').squeeze(1)
        n_batches = (n_ref_imgs - 1) // UPSAMPLE_BATCH + 1
        for b in range(n_batches):
            idx_start = b * UPSAMPLE_BATCH
            idx_end = (b+1) * UPSAMPLE_BATCH
            all_depth[idx_start: idx_end] = net.refine_full(batch.images[ref_idx][idx_start: idx_end],
                                                            all_depth[idx_start: idx_end].unsqueeze(1)).squeeze(1)

        all_depth = all_depth.detach().cpu().numpy()

        return all_depth, None, None


if __name__ == '__main__':
    print('Loading model...')
    net = PL3DVNet.load_from_checkpoint('3dvnet_weights/epoch=100-step=60700.ckpt')
    dset_kwargs = {
        'img_size': (256, 320)
    }
    main('3dvnet', process_scene, dset_kwargs, net, depth=True)
