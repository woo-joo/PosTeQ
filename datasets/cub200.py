import os.path
import shutil
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from datasets.utils import download_and_extract_archive
from datasets.vision import VisionDataset



class CUB200(VisionDataset):
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB200.tar.gz'
    base_folder = 'CUB200'


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

        data_file = open(os.path.join(file_path, 'CUB_200_2011', 'images.txt'))
        data_lines = data_file.readlines()

        target_file = open(os.path.join(file_path, 'CUB_200_2011', 'image_class_labels.txt'))
        target_lines = target_file.readlines()

        ids = {'train': [], 'test': []}
        ids_file = open(os.path.join(file_path, 'CUB_200_2011', 'train_test_split.txt'))
        lines = ids_file.readlines()
        for line in lines:
            idx, flag = line.split()
            split = 'train' if flag == '1' else 'test'
            ids[split].append(int(idx)-1)

        for split in ['train', 'test']:
            data, targets = [], []
            for i in ids[split]:
                idx, image_path = data_lines[i].split()
                image = Image.open(os.path.join(file_path, 'CUB_200_2011', 'images', image_path))
                if image.getbands()[0] == 'L':
                    image = image.convert('RGB')
                data.append(np.array(image))

                idx, class_label = target_lines[i].split()
                targets.append(int(class_label)-1)
            
            entry = {'data': data, 'targets': targets}
            file = open(os.path.join(file_path, split), 'wb')
            pickle.dump(entry, file)
        
        if remove_finished:
            shutil.rmtree(os.path.join(file_path, 'CUB_200_2011'))
            os.remove(os.path.join(file_path, 'attributes.txt'))


    def extra_repr(self) -> str:
        return f'Split: {self.split.title()}'
