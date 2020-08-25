import glob
import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, dataset_dir, transforms_=None):
        '''
        Construct a dataset with all images from a dir.

        dataset_dir: str. img folder path
        '''
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(dataset_dir) + '/*.jpg'))
        print('files:', len(self.files))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = img.convert("RGB")
        item = self.transform(img)

        return item

    def __len__(self):
        return len(self.files)


class PairedImageDataset(Dataset):
    def __init__(self, dataset_dir, soft_data_dir, mode='A2B', transforms_=None):
        '''
        Construct a dataset with all images from a dir.

        dataset: str. dataset name
        style: str. 'A2B' or 'B2A'
        '''
        self.transform = transforms.Compose(transforms_)
        
        if mode in ['A2B']:
            path_A = os.path.join(dataset_dir, 'train', 'A')
            path_B = os.path.join(soft_data_dir, 'B')
            self.files_A = sorted(glob.glob(path_A + '/*.jpg'))
            self.files_B = sorted(glob.glob(path_B + '/*.png'))
        else:
            path_A = os.path.join(soft_data_dir, 'A')
            path_B = os.path.join(dataset_dir, 'train', 'B')
            self.files_A = sorted(glob.glob(path_A + '/*.png'))
            self.files_B = sorted(glob.glob(path_B + '/*.jpg'))
        print('files_A:', len(self.files_A))
        print('files_B:', len(self.files_B))
        assert len(self.files_A) == len(self.files_B)

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_A = img_A.convert("RGB")
        img_A= np.asarray(img_A) # PIL.Image to np.ndarray
        img_A = np.flip(img_A, axis=1) # data augumentation: horrizental flip
        img_A = Image.fromarray(np.uint8(img_A)) # np.ndarray to PIL.Image
        item_A = self.transform(img_A)

        img_B = Image.open(self.files_B[index % len(self.files_B)])
        img_B = img_B.convert("RGB")
        img_B= np.asarray(img_B) # PIL.Image to np.ndarray
        img_B = np.flip(img_B, axis=1) # data augumentation: horrizental flip
        img_B = Image.fromarray(np.uint8(img_B)) # np.ndarray to PIL.Image
        item_B = self.transform(img_B)
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return len(self.files_A)
