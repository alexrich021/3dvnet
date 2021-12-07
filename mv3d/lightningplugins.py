from pytorch_lightning.plugins import ddp_plugin
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from torch.utils.data import DataLoader
import torch
from torch_geometric.data import Batch


class LightningListLoader(DataLoader):
    skip_keys = ['collate_fn']

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        dl_args = {k: v for k, v in kwargs.items() if k not in self.skip_keys}
        super(LightningListLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=lambda data_list: data_list, **dl_args)


class TorchGeometricDDPWrapper(LightningDistributedDataParallel):
    def __init__(self, model, device_ids, follow_batch):
        super(TorchGeometricDDPWrapper, self).__init__(model, device_ids, find_unused_parameters=True)
        self.follow_batch = follow_batch

    def scatter(self, data_list, kwargs, device_ids):
        data_list, batch_idx = data_list
        n_devices = min(len(device_ids), len(data_list))
        count = len(data_list)
        n_per_device = (count - 1) // n_devices + 1
        inputs = [
            (
                Batch.from_data_list(data_list[n_per_device * i: n_per_device * (i + 1)],
                                     follow_batch=self.follow_batch).to(torch.device('cuda:{}'.format(device_ids[i]))),
                batch_idx
            )
            for i in range(n_devices)
        ]
        kwargs = ({}, ) * len(inputs)
        return inputs, kwargs


class TorchGeometricDDPPlugin(ddp_plugin.DDPPlugin):
    def __init__(self, **kwargs):
        super(TorchGeometricDDPPlugin, self).__init__(**kwargs)

    def configure_ddp(self, model, device_ids):
        model = TorchGeometricDDPWrapper(model, device_ids, **self._ddp_kwargs)
        return model
