import numpy as np
import open3d as o3d
import torch


def calc_2d_depth_metrics_batched(depth_pred, depth_gt, pred_valid=None, batch_size=100):
    n_imgs = depth_pred.shape[0]
    n_batches = (n_imgs - 1) // batch_size + 1
    mets = []
    n = []
    for b in range(n_batches):
        idx_start = b * batch_size
        idx_end = (b+1)*batch_size
        pred = depth_pred[idx_start: idx_end]
        gt = depth_gt[idx_start: idx_end]
        valid = None if pred_valid is None else pred_valid[idx_start: idx_end]
        mets.append(calc_2d_depth_metrics(pred, gt, valid, True))
        n.append(pred.shape[0])
    n_sum = float(np.sum(n))
    mets_avg = {}
    for k in mets[0].keys():
        mets_avg[k] = np.sum([n[j]*m[k] for j, m in enumerate(mets)]) / n_sum
    return mets_avg


def calc_2d_depth_metrics(depth_pred, depth_gt, pred_valid=None, convert_to_cpu=False):
    out = {}
    with torch.no_grad():
        valid = (depth_gt >= 0.5) & (depth_gt < 65.)
        if pred_valid is not None:
            valid = valid & pred_valid
            v_perc = torch.mean(torch.sum(pred_valid, dim=(1, 2)) / (pred_valid.shape[1] * pred_valid.shape[2]))
            out['perc_valid'] = v_perc
        valid = valid.type(torch.float)
        denom = torch.sum(valid, dim=(1, 2)) + 1e-7
        abs_diff = torch.abs(depth_pred - depth_gt)
        abs_inv = torch.abs(1. / depth_pred - 1. / depth_gt)
        abs_inv[torch.isinf(abs_inv)] = 0.  # handle div/0
        abs_inv[torch.isnan(abs_inv)] = 0.

        abs_rel = torch.mean(torch.sum((abs_diff / (depth_gt + 1e-7)) * valid, dim=(1, 2)) / denom)
        sq_rel = torch.mean(torch.sum((abs_diff ** 2 / (depth_gt + 1e-7)) * valid, dim=(1, 2)) / denom)
        rmse = torch.mean(torch.sqrt(torch.sum(abs_diff ** 2 * valid, dim=(1, 2)) / denom))
        abs_diff = torch.mean(torch.sum(abs_diff * valid, dim=(1, 2)) / denom)
        abs_inv = torch.mean(torch.sum(abs_inv * valid, dim=(1, 2)) / denom)

        r1 = (depth_pred / depth_gt).unsqueeze(-1)
        r2 = (depth_gt / depth_pred).unsqueeze(-1)
        rel_max = torch.max(torch.cat((r1, r2), dim=-1), dim=-1)[0]

        d_125 = torch.mean(torch.sum((rel_max < 1.25) * valid, dim=(1, 2)) / denom)
        d_125_2 = torch.mean(torch.sum((rel_max < 1.25 ** 2) * valid, dim=(1, 2)) / denom)
        d_125_3 = torch.mean(torch.sum((rel_max < 1.25 ** 3) * valid, dim=(1, 2)) / denom)

        out.update({
            'abs_rel': abs_rel,
            'abs_diff': abs_diff,
            'abs_inv': abs_inv,
            'sq_rel': sq_rel,
            'rmse': rmse,
            'd_125': d_125,
            'd_125_2': d_125_2,
            'd_125_3': d_125_3
        })
        if convert_to_cpu:
            out = {k: v.cpu().item() for k, v in out.items()}
    return out


def eval_mesh(pcd_pred, pcd_trgt, threshold=.05):
    """ Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal

    Returns:
        Dict of mesh metrics
    """

    _, dist1 = nn_correspondance(pcd_trgt, pcd_pred)
    _, dist2 = nn_correspondance(pcd_pred, pcd_trgt)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist1<threshold).astype('float'))
    recal = np.mean((dist2<threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal + 1e-8)
    metrics = {
        'acc': np.mean(dist1),
        'comp': np.mean(dist2),
        'prec': precision,
        'recal': recal,
        'fscore': fscore,
    }
    return metrics


def nn_correspondance(pcd1, pcd2):
    """ for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(pcd1.points) == 0 or len(pcd2.points) == 0:
        return indices, distances
    kdtree = o3d.geometry.KDTreeFlann(pcd1)

    for vert in np.asarray(pcd2.points):
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances