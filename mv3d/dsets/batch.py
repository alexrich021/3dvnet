from torch_geometric import data
import numpy as np
import torch


class Batch(data.Data):
    '''
    Batch class used in both the suncg dataloader and the scannet dataloader
    '''
    def __init__(self, images, rotmats, tvecs, K, depth_images, ref_src_edges):
        super(Batch, self).__init__()
        self.images = images
        self.rotmats = rotmats
        self.tvecs = tvecs
        self.K = K
        self.depth_images = depth_images
        self.ref_src_edges = ref_src_edges

    def __inc__(self, key, value):
        if key == 'ref_src_edges':
            return self.images.shape[0]
        else:
            return super(Batch, self).__inc__(key, value)

    def __cat_dim__(self, key, value):
        if 'edges' in key:
            return 1
        else:
            return 0

    def save(self, filepath):
        np.savez(
            filepath,
            images=self.images.detach().cpu().numpy(),
            rotmats=self.rotmats.detach().cpu().numpy(),
            tvecs=self.tvecs.detach().cpu().numpy(),
            K=self.K.detach().cpu().numpy(),
            depth_images=self.depth_images.detach().cpu().numpy(),
            ref_src_edges=self.ref_src_edges.detach().cpu().numpy(),
        )

    @staticmethod
    def load(filepath):
        data = np.load(filepath)
        return Batch(
            images=torch.from_numpy(data['images']).float(),
            rotmats=torch.from_numpy(data['rotmats']).float(),
            tvecs=torch.from_numpy(data['tvecs']).float(),
            K=torch.from_numpy(data['K']).float(),
            depth_images=torch.from_numpy(data['depth_images']).float(),
            ref_src_edges=torch.from_numpy(data['ref_src_edges']).long(),
        )
