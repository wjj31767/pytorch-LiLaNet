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

class DENSE(data.Dataset):
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
        Class('clear', 100, (0, 0, 142)),
        Class('rain', 101, (220, 20, 60)),
        Class('fog', 102, (119, 11, 32)),
    ]

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.lidar_path = self.root
        self.split = os.path.join(self.root, 'ImageSet', '{}.txt'.format(split))
        self.transform = transform
        self._cache = os.path.join(self.root,"lidar_2d_cache")
        if split not in ['train', 'val']:
            raise ValueError('Invalid split! Use split="train", split="val" or split="all"')
        if not os.path.exists(self._cache):
            os.makedirs(self._cache)
        if not os.path.exists(osp.join(self._cache, split)):
            i = 0
            with lmdb.open(
                    osp.join(self._cache, split), map_size=1 << 35
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                for directionary in os.listdir(self.root):
                    if split in directionary:
                        for classfold in os.listdir(osp.join(self.root,directionary)):
                            print("processing",directionary,classfold)
                            for file in tqdm.tqdm(os.listdir(osp.join(self.root,directionary,classfold))):
                                with h5py.File(osp.join(self.root,directionary,classfold,file), "r") as f:
                                    point_set = np.hstack((np.array(f["sensorX_1"]), np.array(f["sensorY_1"]),np.array(f["sensorZ_1"]), np.array(f['distance_m_1']),np.array(f['intensity_1']), np.array(f['labels_1'])))
                                    txn.put(
                                        str(i).encode(),
                                        msgpack_numpy.packb(
                                            dict(pc=point_set), use_bin_type=True
                                        ),
                                    )
                                i+=1
        self._lmdb_file = osp.join(self._cache, split)
        with lmdb.open(self._lmdb_file, map_size=1 << 35) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = None


    def __getitem__(self, index):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_file, map_size=1 << 35, readonly=True, lock=False
            )

        with self._lmdb_env.begin(buffers=True) as txn:
            ele = msgpack_numpy.unpackb(txn.get(str(index).encode()), raw=False)

        record = ele["pc"].reshape(32,400,6)
        record = torch.from_numpy(record.copy()).permute(2, 0, 1).contiguous()

        distance = record[3, :, :]
        reflectivity = record[4, :, :]
        label = record[5, :, :]

        if self.transform:
            distance, reflectivity, label = self.transform(distance, reflectivity, label)

        return distance, reflectivity, label

    def __len__(self):
        return self._len

    @staticmethod
    def num_classes():
        return len(DENSE.classes)

    @staticmethod
    def mean():
        return [0.21, 12.12]

    @staticmethod
    def std():
        return [0.16, 12.32]

    @staticmethod
    def class_weights():
        return torch.tensor([1 / 15.0, 1.0, 10.0, 10.0])

    @staticmethod
    def get_colormap():
        cmap = torch.zeros([256, 3], dtype=torch.uint8)

        for cls in DENSE.classes:
            cmap[cls.id, :] = torch.tensor(cls.color, dtype=torch.uint8)

        return cmap


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    joint_transforms = Compose([
        RandomHorizontalFlip(),
        Normalize(mean=DENSE.mean(), std=DENSE.std())
    ])


    def _normalize(x):
        return (x - x.min()) / (x.max() - x.min())


    def visualize_seg(label_map, one_hot=False):
        if one_hot:
            label_map = np.argmax(label_map, axis=-1)

        out = np.zeros((label_map.shape[0], label_map.shape[1], 3))

        for l in range(1, DENSE.num_classes()):
            mask = label_map == l
            out[mask, 0] = np.array(DENSE.classes[l].color[1])
            out[mask, 1] = np.array(DENSE.classes[l].color[0])
            out[mask, 2] = np.array(DENSE.classes[l].color[2])

        return out


    dataset = DENSE('../../data/dense', transform=joint_transforms)
    print(dataset.__len__())
    distance, reflectivity, label = random.choice(dataset)

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