import os.path
from scipy.io import loadmat
import shutil
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from datasets.utils import download_and_extract_archive, download_url
from datasets.vision import VisionDataset



class Flowers102(VisionDataset):
    url_prefix = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/'
    url = {'image': url_prefix + '102flowers.tgz',
           'label': url_prefix + 'imagelabels.mat',
           'setid': url_prefix + 'setid.mat'}
    filename = {'image': 'Flowers102_image.tgz',
                'label': 'Flowers102_label.mat',
                'setid': 'Flowers102_setid.mat'}
    base_folder = 'Flowers102'


    def __init__(
        self,
        root: str,
        split: bool = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # training set or test set

        if download:
            self.download()

        if not self.check_file():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        self.data: Any = []
        self.targets = []

        file_path = os.path.join(self.root, self.base_folder, self.split)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f)
            self.data = entry['data']
            self.targets = entry['targets']


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)


    def check_file(self) -> bool:
        return os.path.isfile(os.path.join(self.root, self.base_folder, self.split))


    def download(self) -> None:
        if self.check_file():
            print('Files already downloaded and verified')
            return
        
        extract_root = os.path.join(self.root, self.base_folder)
        download_and_extract_archive(self.url['image'], self.root, extract_root=extract_root, filename=self.filename['image'], remove_finished=True)
        download_url(self.url['label'], extract_root, filename=self.filename['label'])
        download_url(self.url['setid'], extract_root, filename=self.filename['setid'])
        self.make_train_test(remove_finished=True)


    def make_train_test(self, remove_finished: bool = False) -> None:
        file_path = os.path.join(self.root, self.base_folder)

        targets = loadmat(os.path.join(file_path, self.filename['label']), squeeze_me=True)['labels'] - 1
        setids = loadmat(os.path.join(file_path, self.filename['setid']), squeeze_me=True)
        ids = {'train': np.sort(np.concatenate((setids['trnid'], setids['valid']))) - 1,
               'test' : np.sort(setids['tstid']) - 1}
        
        for split in ['train', 'test']:
            data = []
            for i in ids[split]:
                image_path = os.path.join(file_path, 'jpg/image_{0:05d}.jpg'.format(i+1))
                image = Image.open(image_path)
                data.append(np.array(image))

            entry = {'data': data, 'targets': targets[ids[split]]}
            file = open(os.path.join(file_path, split), 'wb')
            pickle.dump(entry, file)

        if remove_finished:
            shutil.rmtree(os.path.join(file_path, 'jpg'))
            os.remove(os.path.join(file_path, self.filename['label']))
            os.remove(os.path.join(file_path, self.filename['setid']))


    def extra_repr(self) -> str:
        return f'Split: {self.split.title()}'
