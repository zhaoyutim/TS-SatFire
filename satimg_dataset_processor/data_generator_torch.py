import os

import numpy as np
import torch
from monai.metrics import MeanIoU, DiceMetric
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample, label, augmentation=True):
        if augmentation:
            hflip = bool(np.random.random() > 0.5)
            vflip = bool(np.random.random() > 0.5)
            rotate = int(np.floor(np.random.random() * 4))
            if hflip:
                sample = TF.hflip(sample)
                label = TF.hflip(label)

            if vflip:
                sample = TF.vflip(sample)
                label = TF.vflip(label)

            if rotate != 0:
                angle = rotate * 90
                sample = TF.rotate(sample, angle)
                label = TF.rotate(label, angle)

        for i in range(len(self.mean)):
            sample[i, :, ...] = (sample[i, :, ...] - self.mean[i]) / self.std[i]
        return sample, label

class FireDataset(Dataset):
    def __init__(self, image_path, label_path, ts_length=8, transform=None, n_channel=8, label_sel=0):
        self.image_path, self.label_path = image_path, label_path
        self.num_samples = np.load(self.image_path).shape[0]
        self.transform = transform
        self.n_channel = n_channel
        self.label_sel = label_sel
        self.ts_length = ts_length
        if 'train' in self.label_path:
            self.augmentation = True
        else:
            self.augmentation = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load a chunk of data from disk
        data_chunk, label_chunk = self.load_data(idx)
        if self.transform:
            data_chunk, label_chunk = self.transform(data_chunk, label_chunk, self.augmentation)
        sample = {
            'data': data_chunk,
            'labels': label_chunk,
        }

        return sample

    # define a function to load a batch of data from disk
    def load_data(self, indices):
        # load a chunk of data from disk
        data_chunk = np.load(self.image_path, mmap_mode='r')[indices]
        label_chunk = np.load(self.label_path, mmap_mode='r')[indices]

        if self.n_channel==6:
            img_dataset = data_chunk[2:, :, :, :]
        else:
            img_dataset = data_chunk[:, :, :, :]
        label_dataset = label_chunk[[self.label_sel], :, :, :]
        # 0 NIFC 1 VIIRS AF ACC 2 combine
        y_dataset = np.zeros((2, self.ts_length, 256,256))
        # y_dataset = np.where(label_dataset > 0, 1, 0)
        # y_dataset = np.where(af_dataset > 0, 2, y_dataset)
        y_dataset[0, :, :, :] = label_dataset[..., :] == 0
        y_dataset[1, :, :, :] = label_dataset[..., :] > 0

        x_array, y_array = img_dataset, y_dataset
        x_array_copy = x_array.copy()
        # convert the data to a PyTorch tensor
        x = torch.squeeze(torch.from_numpy(x_array_copy))
        y = torch.squeeze(torch.from_numpy(y_array)).long()


        return x, y

if __name__ == '__main__':
    root_path = '/home/z/h/zhao2/TS-SatFire/dataset/'
    mode = 'ba'
    interval = 3
    ts_length = 6
    image_path = os.path.join(root_path, 'dataset_val/'+mode+'_val_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
    label_path = os.path.join(root_path, 'dataset_val/'+mode+'_val_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
    transform = Normalize(mean = [18.76488,27.441864,20.584806,305.99478,294.31738,14.625097,276.4207,275.16766],
                        std = [15.911591,14.879259,10.832616,21.761852,24.703484,9.878246,40.64329,40.7657])
    train_dataset = FireDataset(image_path=image_path, label_path=label_path, transform=transform, ts_length=ts_length, n_channel=8)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    print(next(iter(train_dataloader)).get('data').shape)