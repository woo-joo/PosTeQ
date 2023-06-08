import os.path
import shutil
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from datasets.utils import download_and_extract_archive
from datasets.vision import VisionDataset



class CIFAR100(VisionDataset):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = 'CIFAR100.tar.gz'
    base_folder = 'CIFAR100'


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
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))


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
        download_and_extract_archive(self.url, self.root, extract_root=extract_root, filename=self.filename, remove_finished=True)
        self.make_train_test(remove_finished=True)
    

    def make_train_test(self, remove_finished: bool = False) -> None:
        file_path = os.path.join(self.root, self.base_folder)
        files = os.listdir(os.path.join(file_path, 'cifar-100-python'))
        for file in files:
            if file in ['train', 'test']:
                src_path = os.path.join(file_path, 'cifar-100-python', file)
                dst_path = os.path.join(file_path, file)
                os.rename(src_path, dst_path)
        
        if remove_finished:
            shutil.rmtree(os.path.join(file_path, 'cifar-100-python'))


    def extra_repr(self) -> str:
        return f'Split: {self.split.title()}'
