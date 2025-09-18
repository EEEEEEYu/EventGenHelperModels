import os
import weakref
import numpy as np
import h5py
import yaml
import cv2
import torch
import imageio.v2 as imageio
import torchvision.transforms as T
from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import Dataset
from utils.eventslicer import EventSlicer

def load_flow_map16(path):
    """
    Load a 3-channel 16-bit PNG optical flow:
      - channel 0 (R): x-disp  
      - channel 1 (G): y-disp  
      - channel 2 (B): valid flag  

    Conversion to float:
      flow = (raw − 2**15) / 128.0
    """
    raw = imageio.imread(path, format='PNG-FI')
    raw = np.asarray(raw, dtype=np.uint16)
    r = raw[..., 0].astype(np.int32)
    g = raw[..., 1].astype(np.int32)
    b = raw[..., 2]

    flow_x = (r - 2**15) / 128.0
    flow_y = (g - 2**15) / 128.0
    valid  = (b > 0)

    return flow_x, flow_y, valid

class Dsec(Dataset):
    def __init__(self, seq_path, task_name, purpose='train', delta_t_ms=50, num_bins=15, chunk_size=4):
        self.seq_path = seq_path

        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code if duration >100 ms'
        assert os.path.isdir(seq_path), f"{seq_path!r} is not a directory"

        self.task_name = task_name
        self.purpose    = purpose
        self.height     = 480
        self.width      = 640
        self.num_bins   = num_bins
        self.location   = 'left'
        self.delta_t_us = delta_t_ms * 1000
        self.direction  = 'forward'
        self.chunk_size = chunk_size
        self.transform = T.Compose([
            T.ToTensor(), # Automatically transforms numpy array of shape (H, W, C) into tensor of shape (C, H, W) and normalize into [0.0, 1.0]
        ])

        # ── optical-flow timestamps & paths ──
        flow_base = os.path.join(seq_path, 'flow')
        flow_ts_file   = os.path.join(flow_base, 'forward_timestamps.txt')
        assert os.path.isfile(flow_ts_file), f"Missing {flow_ts_file}"
        self.flow_timestamps = np.loadtxt(flow_ts_file, delimiter=',', dtype=np.int64)

        pngs = [
            os.path.join(flow_base, self.direction, f)
            for f in os.listdir(os.path.join(flow_base, self.direction))
            if f.endswith('.png')
        ]
        self.flow_paths = sorted(pngs,
                                 key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        assert len(self.flow_paths) == len(self.flow_timestamps), (
            f"{len(self.flow_paths)} images vs {len(self.flow_timestamps)} timestamps"
        )

        # ── event data ──
        ev_base = os.path.join(seq_path, 'event', self.location)
        ev_file = os.path.join(ev_base, 'events.h5')
        rm_file = os.path.join(ev_base, 'rectify_map.h5')
        assert os.path.isfile(ev_file), f"Missing {ev_file}"
        assert os.path.isfile(rm_file), f"Missing {rm_file}"

        # ── image timestamps & paths ──
        image_base = os.path.join(seq_path, 'image')
        image_ts_file    = os.path.join(image_base, 'timestamps.txt')
        exposure_ts_file = os.path.join(image_base, 'exposure_timestamps_left.txt')
        assert os.path.isfile(image_ts_file), f"Missing {image_ts_file}"
        assert os.path.isfile(exposure_ts_file), f"Missing {exposure_ts_file}"
        self.image_timestamps = np.loadtxt(image_ts_file, dtype=np.int64)
        self.exposure_timestamps = np.loadtxt(exposure_ts_file, delimiter=',', dtype=np.int64)

        png_files = [
            os.path.join(image_base, self.location, fn)
            for fn in os.listdir(os.path.join(image_base, self.location)) if fn.endswith('.png')
        ]
        self.image_paths = sorted(
            png_files,
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
        assert len(self.image_paths) == len(self.image_timestamps), (
            f"{len(self.image_paths)} images vs {len(self.image_timestamps)} timestamps"
        )

        # ── load calibration ──
        calib_file = os.path.join(seq_path, 'calibration', 'cam_to_cam.yaml')
        assert os.path.isfile(calib_file), f"Missing calib: {calib_file}"
        with open(calib_file, 'r') as f:
            conf = yaml.safe_load(f)

        # rectified intrinsics
        fx0, fy0, cx0, cy0 = conf['intrinsics']['camRect0']['camera_matrix']
        K_r0 = np.array([[fx0,  0, cx0],
                         [  0, fy0, cy0],
                         [  0,   0,   1]], dtype=np.float32)
        fx1, fy1, cx1, cy1 = conf['intrinsics']['camRect1']['camera_matrix']
        K_r1 = np.array([[fx1,  0, cx1],
                         [  0, fy1, cy1],
                         [  0,   0,   1]], dtype=np.float32)

        # rotations & extrinsic
        R0 = np.array(conf['extrinsics']['R_rect0'])
        R1 = np.array(conf['extrinsics']['R_rect1'])
        R10 = np.array(conf['extrinsics']['T_10'])[:3, :3]

        # infinite-depth homography
        R_r1_r0 = R1 @ R10 @ R0.T
        H_inf    = K_r1 @ R_r1_r0 @ np.linalg.inv(K_r0)

        # precompute remap grid
        ht, wd = self.height, self.width
        xs, ys = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
        proj   = H_inf.dot(coords.reshape(-1,3).T).T
        proj  /= proj[:, 2:3]
        self.image_warp_map = proj[:,:2].astype(np.float32).reshape(ht, wd, 2)

        # ── streams & finalizer ──
        self.h5f          = {self.location: h5py.File(ev_file, 'r')}
        self.event_slicer = EventSlicer(self.h5f[self.location])
        with h5py.File(rm_file, 'r') as rm:
            self.rectify_ev_maps = {self.location: rm['rectify_map'][()]}
        self._finalizer = weakref.finalize(self, self._cleanup, self.h5f)

        self.image_idx_offset = self.get_image_idx_offset()

    def __len__(self):
        return self.chunk_size
        # return len(self.image_timestamps) - 1 if self.task_name == 'e2rgb' else len(self.flow_timestamps)

    def rectify_events(self, x, y):
        rm = self.rectify_ev_maps[self.location]
        assert x.max() < self.width
        assert y.max() < self.height
        assert rm.shape == (self.height, self.width, 2), rm.shape
        return rm[y, x]
    
    # In event-to-RGB task, calculate the image offset so that the first frame has full event info after it.
    def get_image_idx_offset(self):
        event_start_time = self.event_slicer.get_start_time_us()
        image_idx_offset = 0
        while self.exposure_timestamps[image_idx_offset][1] < event_start_time:
            image_idx_offset += 1
        return image_idx_offset

    def __getitem__(self, idx):
        #                                 f0   f1                f2   f3
        # rgb     |    |        e0        |    |        e1       |    |     
        # events     --------------------------------------------------------- 
        # idx:    0 1 2 3 4 5 6 7  8
        # anchor: 2 2 2 2 6 6 6 6  10
        # next:   3 4 5 6 7 8 9 10 11
        if self.task_name == 'e2rgb':
            # The first RGB image is used for cold-start gaussian training, no events info involved
            if idx <= self.image_idx_offset:
                _, f0 = self.exposure_timestamps[self.image_idx_offset]
                f1 = f0
                rectified_xy_filtered = torch.empty(0)
                p = torch.empty(0)
                t = torch.empty(0)
            # The rest event + RGB pair is used for regular 4D gaussian training
            else:
                _, f0 = self.exposure_timestamps[max(self.image_idx_offset - 1 + (idx // self.chunk_size) * self.chunk_size, 0)]
                _, f1 = self.exposure_timestamps[self.image_idx_offset + idx]
                ev = self.event_slicer.get_events(f0, f1)
                
                if ev is None:
                    raise ValueError(f"No events in between timestamps {f0} and {f1}")

                p, x, y, t = ev['p'], ev['x'], ev['y'], ev['t']
                xy_r = torch.from_numpy(self.rectify_events(x, y))

                frac = xy_r - torch.floor(xy_r)
                rectified_xy_rounded = torch.where(frac <= 0.5, torch.floor(xy_r), torch.ceil(xy_r)).long()
                rectified_x = rectified_xy_rounded[:, 0]
                rectified_y = rectified_xy_rounded[:, 1]

                mask = (rectified_x >= 0) & (rectified_x < self.width) & \
                    (rectified_y >= 0) & (rectified_y < self.height)

                rectified_xy_filtered = rectified_xy_rounded[mask]
                p = p[mask]
                t = t[mask]

            # RGB image calibration
            img = cv2.imread(self.image_paths[idx + self.image_idx_offset], cv2.IMREAD_UNCHANGED)
            assert img is not None, "Failed to load image"
            warped  = cv2.remap(
                img,
                self.image_warp_map[...,0],
                self.image_warp_map[...,1],
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT
            )
            warped = self.transform(warped)

            sample = {
                'timestamp': (f0, f1),
                'events':    {'p': p, 'xy': rectified_xy_filtered, 't': t},
                'image':   {'path': self.image_paths[idx + self.image_idx_offset], 'data': warped},
                'seq_path': self.seq_path
            }

        elif self.task_name == 'flow_ssl':
            
            t0, t1 = self.flow_timestamps[idx]

            # slice & rectify events
            ev = self.event_slicer.get_events(t0, t1)
            p, x, y, t = ev['p'], ev['x'], ev['y'], ev['t']
            xy_r = torch.from_numpy(self.rectify_events(x, y))

            frac = xy_r - torch.floor(xy_r)
            rectified_xy_rounded = torch.where(frac <= 0.5, torch.floor(xy_r), torch.ceil(xy_r)).long()
            rectified_x = rectified_xy_rounded[:, 0]
            rectified_y = rectified_xy_rounded[:, 1]

            mask = (rectified_x >= 0) & (rectified_x < self.width) & \
                (rectified_y >= 0) & (rectified_y < self.height)

            rectified_xy_filtered = rectified_xy_rounded[mask]
            p = p[mask]
            t = t[mask]

            fx, fy, valid = load_flow_map16(self.flow_paths[idx])

            # slice & rectify events + flow
            sample = {
                'timestamp': (t0, t1),
                'events':    {'p': p, 'xy': rectified_xy_filtered, 't': t},
                'flow':      {'x': fx, 'y': fy, 'valid': valid}
            }

            # load & warp the left frame into the event frame
            img_idx = np.searchsorted(self.image_timestamps, t0, side='left') - 1
            img     = cv2.imread(self.image_paths[img_idx], cv2.IMREAD_UNCHANGED)
            assert img is not None, "Failed to load image"
            warped  = cv2.remap(
                img,
                self.image_warp_map[...,0],
                self.image_warp_map[...,1],
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT
            )
            warped = self.transform(warped)
            sample['image'] = {'path': self.image_paths[img_idx], 'data': warped}

        return sample

    @staticmethod
    def _cleanup(h5f_dict):
        for f in h5f_dict.values():
            f.close()


if __name__ == '__main__':
    # Example usage
    dataset = Dsec(seq_path='/fs/nexus-scratch/tuxunlu/git/E4DGaussian/dataset/zurich_city_02_a', task_name='flow_ssl', purpose='train')
    # enumerate dataset
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Timestamp: {sample['timestamp']}")
        print(f"  Events: {sample['events']}")
        print(f"  Image path: {sample['image']['path']}")
        print(f"  Image shape: {sample['image']['data'].shape}")
        if 'flow' in sample:
            print(f"  Flow x shape: {sample['flow']['x'].shape}")
            print(f"  Flow y shape: {sample['flow']['y'].shape}")
            print(f"  Flow valid shape: {sample['flow']['valid'].shape}")