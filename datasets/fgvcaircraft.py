import os.path
from scipy.io import loadmat
import shutil
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from datasets.utils import download_and_extract_archive, download_url
from datasets.vision import VisionDataset



class FGVCAircraft(VisionDataset):
    url = 'https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    filename = 'FGVCAircraft.tar.gz'
    base_folder = 'FGVCAircraft'


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
        download_and_extract_archive(self.url, self.root, extract_root=extract_root, filename=self.filename, remove_finished=True)
        self.make_train_test(remove_finished=True)


    def make_train_test(self, remove_finished: bool = False) -> None:
        file_path = os.path.join(self.root, self.base_folder)

        target_file = open(os.path.join(file_path, 'fgvc-aircraft-2013b/data/variants.txt'))
        target_lines = target_file.readlines()
        targets_dict = {target_name[:-1]: target for target, target_name in enumerate(target_lines)}

        split_dict = {'train': 'trainval', 'test': 'test'}
        for split in ['train', 'test']:
            data_file = open(os.path.join(file_path, f'fgvc-aircraft-2013b/data/images_variant_{split_dict[split]}.txt'))
            data_lines = data_file.readlines()

            data, targets = [], []
            for line in data_lines:
                image_idx, target_name = line.split(' ', 1)
                image = Image.open(os.path.join(file_path, f'fgvc-aircraft-2013b/data/images/{image_idx}.jpg'))
                data.append(np.array(image))
                targets.append(targets_dict[target_name[:-1]])
            
            entry = {'data': data, 'targets': targets}
            file = open(os.path.join(file_path, split), 'wb')
            pickle.dump(entry, file)

        if remove_finished:
            shutil.rmtree(os.path.join(file_path, 'fgvc-aircraft-2013b'))


    def extra_repr(self) -> str:
        return f'Split: {self.split.title()}'
