import numpy as np


class FrameSelector:
    def __init__(self):
        ...

    def select_frames(self, poses, n_frames, seed_idx=None):
        ...


class RangePoseDistSelector(FrameSelector):
    def __init__(self, p_min, p_max, search_interval):
        super(RangePoseDistSelector, self).__init__()
        self.p_min = p_min
        self.p_max = p_max
        self.p_opt = p_min + (p_max-p_min) / 2.
        self.search_interval = search_interval

    def select_frames(self, poses, n_frames, seed_idx=None):
        n_frames_total = poses.shape[0]
        max_idx = n_frames_total - n_frames * self.search_interval - 1

        if seed_idx is None:
            if max_idx <= 0:
                seed_idx = 0
            else:
                seed_idx = np.random.randint(0, max_idx)

        img_idx = [seed_idx, ]
        for i in range(n_frames - 1):
            pdists = []
            P_ref_inv = np.linalg.inv(poses[img_idx[i]])
            for j in range(self.search_interval):
                src_idx = img_idx[i] + j + 1
                if src_idx >= n_frames_total:
                    break

                P_src = poses[src_idx]
                P_rel = P_ref_inv @ P_src
                R_rel = P_rel[:3, :3]
                t_rel = P_rel[:3, 3, None]
                pdists.append(np.sqrt(np.sum(t_rel ** 2) + (2. / 3.) * np.trace(np.eye(3, dtype=np.float32) - R_rel)))
            if len(pdists) == 0:
                break
            pdists = np.asarray(pdists)
            valid_idx = (pdists > self.p_min) & (pdists < self.p_max)
            if np.sum(valid_idx) == 0:
                idx = np.argmin(np.abs(pdists - self.p_opt))  # select best pose
            else:
                idx = np.random.choice(np.arange(len(pdists))[valid_idx])  # select random pose pdist range
            img_idx.append(img_idx[i] + idx + 1)
        img_idx = np.asarray(img_idx)
        return img_idx


class BestPoseDistSelector(FrameSelector):
    def __init__(self, p_opt, search_interval):
        super(BestPoseDistSelector, self).__init__()
        self.p_opt = p_opt
        self.search_interval = search_interval

    def select_frames(self, poses, n_frames, seed_idx=None):
        n_frames_total = poses.shape[0]
        max_idx = n_frames_total - n_frames * self.search_interval - 1

        if seed_idx is None:
            if max_idx <= 0:
                seed_idx = 0
            else:
                seed_idx = np.random.randint(0, max_idx)

        img_idx = [seed_idx, ]
        for i in range(n_frames - 1):
            pdists = []
            P_ref_inv = np.linalg.inv(poses[img_idx[i]])
            for j in range(self.search_interval):
                src_idx = img_idx[i] + j + 1
                if src_idx >= n_frames_total:
                    break

                P_src = poses[src_idx]
                P_rel = P_ref_inv @ P_src
                R_rel = P_rel[:3, :3]
                t_rel = P_rel[:3, 3, None]
                pdists.append(np.sqrt(np.sum(t_rel ** 2) + (2. / 3.) * np.trace(np.eye(3, dtype=np.float32) - R_rel)))
            if len(pdists) == 0:
                break
            pdists = np.asarray(pdists)
            idx = np.argmin(np.abs(pdists - self.p_opt))  # select best pose
            img_idx.append(img_idx[i] + idx + 1)
        img_idx = np.asarray(img_idx)
        return img_idx


class NextPoseDistSelector(FrameSelector):
    def __init__(self, p_thresh, search_interval=30):
        super(NextPoseDistSelector, self).__init__()
        self.p_thresh = p_thresh
        self.search_interval = search_interval

    def select_frames(self, poses, n_frames, seed_idx=None):
        n_frames_total = poses.shape[0]
        max_idx = n_frames_total - n_frames * self.search_interval - 1

        if seed_idx is None:
            if max_idx <= 0:
                seed_idx = 0
            else:
                seed_idx = np.random.randint(0, max_idx)

        img_idx = [seed_idx, ]
        for i in range(n_frames - 1):
            P_ref_inv = np.linalg.inv(poses[img_idx[i]])
            current_idx = img_idx[-1] + 1
            for j in range(self.search_interval):
                if current_idx > n_frames_total - 1:
                    break
                P_src = poses[current_idx]
                P_rel = P_ref_inv @ P_src
                R_rel = P_rel[:3, :3]
                t_rel = P_rel[:3, 3, None]
                pdist = np.sqrt(np.sum(t_rel ** 2) + (2. / 3.) * np.trace(np.eye(3, dtype=np.float32) - R_rel))

                if pdist >= self.p_thresh:
                    break
                current_idx += 1

            if current_idx > n_frames_total - 1:
                break
            img_idx.append(current_idx)
        img_idx = np.asarray(img_idx)
        return img_idx


class NeuralReconSelector(FrameSelector):
    def __init__(self, tmin=.1, rmin_deg=15):
        super().__init__()
        self.tmin = tmin
        self.rmin_deg = rmin_deg

    def select_frames(self, poses, n_frames, seed_idx=None):
        cos_t_max = np.cos(self.rmin_deg * np.pi / 180)
        frame_inds = np.arange(len(poses))
        if seed_idx is not None:
            frame_inds = np.roll(frame_inds, seed_idx)
        selected_frame_inds = [frame_inds[0]]
        for frame_ind in frame_inds[1:]:
            prev_pose = poses[selected_frame_inds[-1]]
            candidate_pose = poses[frame_ind]
            cos_t = np.sum(prev_pose[:3, 2] * candidate_pose[:3, 2])
            tdist = np.linalg.norm(prev_pose[:3, 3] - candidate_pose[:3, 3])
            if tdist > self.tmin or cos_t < cos_t_max:
                selected_frame_inds.append(frame_ind)
        return np.asarray(selected_frame_inds)


class EveryNthSelector(FrameSelector):
    def __init__(self, interval):
        super(EveryNthSelector, self).__init__()
        self.interval = interval

    def select_frames(self, poses, n_frames, seed_idx=None):
        n_frames_total = poses.shape[0]

        # first, choose a seed idx if necessary
        max_idx = n_frames_total - n_frames * self.interval - 1
        if seed_idx is None:
            if max_idx <= 0:
                seed_idx = 0
            else:
                seed_idx = np.random.randint(0, max_idx)

        # next, determine the end point of the frame selection and use this to construct your image idx
        end_idx = min(n_frames_total, seed_idx + self.interval*(n_frames-1) + 1)
        img_idx = np.arange(seed_idx, end_idx, self.interval)
        return img_idx
