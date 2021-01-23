import os
import random
from collections import namedtuple
import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import os.path as osp
import tqdm
from lilanet.datasets.transforms import Compose, RandomHorizontalFlip, Normalize
import h5py
from torch.utils.data import Dataset
import numpy as np
import mayavi.mlab as mlab
# import kitti_util as utils
# from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

class Image2ImageDataset(Dataset):

    def __init__(self, ds_a, ds_b):
        self.dataset_a = ds_a
        self.dataset_b = ds_b

    def __len__(self):
        return max(len(self.dataset_a), len(self.dataset_b))

    def __getitem__(self, i):
        dp_a = self.dataset_a[i % len(self.dataset_a)]
        j = random.randint(0, len(self.dataset_b) - 1)
        dp_b = self.dataset_b[j]
        return {
            'A': dp_a,
            'B': dp_b
        }

class CYCLEDENSE(data.Dataset):
    """`KITTI LiDAR`_ Dataset.

    Args:
        root (string): Root directory of the ``lidar_2d`` and ``ImageSet`` folder.
        split (string, optional): Select the split to use, ``train``, ``val`` or ``all``
        transform (callable, optional): A function/transform that  takes in distance, reflectivity
            and target tensors and returns a transformed version.
    """

    Class = namedtuple('Class', ['name', 'id', 'color'])

    classes = [
        Class('unlabeled', 0, (0, 0, 0)),
        Class('clear', 1, (0, 0, 142)),
        Class('rain', 2, (220, 20, 60)),
        Class('fog', 3, (119, 11, 32)),
    ]

    def __init__(self, root, split='val', clss='clear', transform=None):
        self.root = os.path.expanduser(root)
        self.lidar_path = self.root
        self.split = os.path.join(
            self.root, 'ImageSet', '{}.txt'.format(split))
        self.transform = transform
        self._cache = os.path.join(self.root, "cycle_lidar_2d_cache")
        if split not in ['train', 'val']:
            raise ValueError(
                'Invalid split! Use split="train", split="val" or split="all"')
        if clss not in ['clear', 'rain','fog']:
            raise ValueError(
                'Invalid class! Use cls="clear", split="rain" or split="fog"')
        if not os.path.exists(self._cache):
            os.makedirs(self._cache)
        if not os.path.exists(osp.join(self._cache, split+"_"+clss)):
            i = 0
            with lmdb.open(
                    osp.join(self._cache, split+"_"+clss), map_size=1 << 34
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                for directionary in os.listdir(self.root):
                    if split in directionary:
                        for classfold in os.listdir(
                                osp.join(self.root, directionary)):
                            if clss not in classfold.lower():
                                continue
                            print("processing", directionary, classfold)
                            for file in tqdm.tqdm(
                                os.listdir(
                                    osp.join(
                                        self.root,
                                        directionary,
                                        classfold))):
                                with h5py.File(osp.join(self.root, directionary, classfold, file), "r") as f:
                                    label = np.array(f['labels_1'])
                                    label[label == 100] = 1
                                    label[label == 101] = 2
                                    label[label == 102] = 3
                                    point_set = np.dstack(
                                        (np.array(
                                            f["sensorX_1"]), np.array(
                                            f["sensorY_1"]), np.array(
                                            f["sensorZ_1"]), np.array(
                                            f['distance_m_1']), np.array(
                                            f['intensity_1']), label))
                                    txn.put(
                                        str(i).encode(),
                                        msgpack_numpy.packb(
                                            dict(pc=point_set), use_bin_type=True
                                        ),
                                    )
                                i += 1
        self._lmdb_file = osp.join(self._cache, split+"_"+clss)
        with lmdb.open(self._lmdb_file, map_size=1 << 34) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = None

    def __getitem__(self, index):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_file, map_size=1 << 34, readonly=True, lock=False
            )

        with self._lmdb_env.begin(buffers=True) as txn:
            ele = msgpack_numpy.unpackb(
                txn.get(str(index).encode()), raw=False)

        record = ele["pc"].astype(np.float32)
        record = torch.from_numpy(record.copy()).permute(2, 0, 1).contiguous()
        pc = record[:3,:,:]
        distance = record[3, :, :]
        reflectivity = record[4, :, :]
        label = record[5, :, :]
        if self.transform:
            distance, reflectivity, label = self.transform(
                distance, reflectivity, label)
        # out = torch.cat([distance,reflectivity,label],0)
        # preprocessing
        return torch.cat([distance,reflectivity],0)
        # return pc,distance,reflectivity,label
    def __len__(self):
        return self._len

    @staticmethod
    def num_classes():
        return len(CYCLEDENSE.classes)

    @staticmethod
    def mean():
        return [0.5, 0.5, 0.5]

    @staticmethod
    def std():
        return [0.5, 0.5, 0.5]

    @staticmethod
    def class_weights():
        return torch.tensor([1 / 15.0, 1.0, 10.0, 10.0])

    @staticmethod
    def get_colormap():
        cmap = torch.zeros([256, 3], dtype=torch.uint8)

        for cls in CYCLEDENSE.classes:
            cmap[cls.id, :] = torch.tensor(cls.color, dtype=torch.uint8)

        return cmap


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    joint_transforms = Compose([
        RandomHorizontalFlip(),
        Normalize(mean=CYCLEDENSE.mean(), std=CYCLEDENSE.std())
    ])


    def _normalize(x):
        return (x - x.min()) / (x.max() - x.min())


    def visualize_seg(label_map, one_hot=False):
        if one_hot:
            label_map = np.argmax(label_map, axis=-1)

        out = np.zeros((label_map.shape[0], label_map.shape[1], 3))

        for l in range(1, CYCLEDENSE.num_classes()):
            mask = label_map == l
            out[mask, 0] = np.array(CYCLEDENSE.classes[l].color[1])
            out[mask, 1] = np.array(CYCLEDENSE.classes[l].color[0])
            out[mask, 2] = np.array(CYCLEDENSE.classes[l].color[2])

        return out


    dataset = CYCLEDENSE('../../data/dense', clss="clear",transform=joint_transforms)
    pc, distance, reflectivity,label = dataset[303]

    print('Distance size: ', distance.size())
    print('Reflectivity size: ', reflectivity.size())
    print('Label size: ', label.size())

    distance_map = Image.fromarray((255 * _normalize(distance.numpy())).astype(np.uint8))
    reflectivity_map = Image.fromarray((255 * _normalize(reflectivity.numpy())).astype(np.uint8))
    label_map = Image.fromarray((255 * visualize_seg(label.numpy())).astype(np.uint8))

    blend_map = Image.blend(distance_map.convert('RGBA'), label_map.convert('RGBA'), alpha=0.4)

    plt.figure(figsize=(10, 5))
    plt.subplot(221)
    plt.title("Distance")
    plt.imshow(distance_map)
    plt.subplot(222)
    plt.title("Reflectivity")
    plt.imshow(reflectivity_map)
    plt.subplot(223)
    plt.title("Label")
    plt.imshow(label_map)
    plt.subplot(224)
    plt.title("Result")
    plt.imshow(blend_map)

    plt.show()
    pc = pc.reshape(3,-1).transpose(1,0)
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
    )

    # draw points
    mlab.points3d(
        pc[:, 0],
        pc[:, 1],
        pc[:, 2],
        color=None,
        mode="point",
        colormap="gnuplot",
        scale_factor=1,
        figure=fig,
    )

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)
    # draw axis
    axes = np.array(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )
    mlab.view(
        azimuth=180,
        elevation=70,
        focalpoint=[12.0909996, -1.04700089, -2.03249991],
        distance=62.0,
        figure=fig,
    )

    mlab.show()


