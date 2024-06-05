import os

import numpy as np
import torch
from monai.metrics import MeanIoU, DiceMetric
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class Normalize(object):
    def __init__(self, mean, std, dont_normalize_idc):
        self.mean = mean
        self.std = std
        self.dont_normalize_idc = dont_normalize_idc

    def __call__(self, sample):
        # todo: use tensor operations instead of loop
        for i in range(len(self.mean)):
            if i not in self.dont_normalize_idc:
                sample[i, :, ...] = (sample[i, :, ...] - self.mean[i]) / self.std[i]
        return sample

class FireDataset(Dataset):
    def __init__(self, image_path, label_path, ts_length=8, use_augmentations=False, n_channel=8, label_sel=0, target_is_single_day=False):
        self.image_path, self.label_path = image_path, label_path
        self.num_samples = np.load(self.image_path, mmap_mode='r').shape[0]
        self.n_channel = n_channel
        self.label_sel = label_sel
        self.ts_length = ts_length
        self.target_is_single_day = target_is_single_day
        self.use_augmentations = use_augmentations
        self.indices_of_degree_features = [12, 18, 24]
        self.normalizer = Normalize(mean=[18.224253, 26.95519, 20.09066, 318.25967, 308.78717, 14.165086, 291.29214, 288.97382, 5110.5547, 2556.2627, 0.3907487, 3.4994626, 216.23518, 276.5463, 291.8275, 70.32086, 0.0054306216, 10.120554, 175.33012, 1290.8367, -1.5219007, 7.3989105, 7.584937, 1.4395763, 3.306973, 19.259102, 0.0057929577],
                          std=[15.438321, 14.408274, 10.552524, 13.1312475, 12.155249, 9.652911, 12.435288, 8.750125, 2400.766, 1206.8983, 2.37979, 1.6343528, 85.730644, 47.332256, 50.045837, 22.48386, 0.0021515382, 8.429097, 104.73222, 823.01483, 1.9954495, 4.1257873, 26.547232, 1.2017097, 48.207355, 5.4114914, 0.0017134654],
                          dont_normalize_idc=(self.indices_of_degree_features + [21])) # don't normalize degrees or landcover class
        self.one_hot_matrix = torch.eye(17) # used in augmentation
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load a chunk of data from disk, x.shape=[B,C,T,H,W], y.shape=[B,H,W]
        x, y = self.load_data(idx)
        
        x = self.normalizer(x)
        if self.use_augmentations:
            x, y = self.augment(x, y)
        x = self.preprocess(x)
        sample = {
            'data': x,
            'labels': y,
        }
        return sample

    def preprocess(self, x):
        # Transform degrees with sine function
        x[self.indices_of_degree_features, ...] = torch.sin(
            torch.deg2rad(x[self.indices_of_degree_features, ...]))
        
        # Create land cover class one-hot encoding, put it where the land cover integer was
        new_shape = (x.shape[1], x.shape[2], x.shape[3],
                     self.one_hot_matrix.shape[0])
        # -1 because land cover classes start at 1
        landcover_classes_flattened = x[21, ...].long().flatten() - 1
        landcover_encoding = self.one_hot_matrix[landcover_classes_flattened].reshape(
            new_shape).permute(3, 0, 1, 2)
        x = torch.concatenate(
            [x[:21, ...], landcover_encoding, x[22:, ...]], dim=0)
        return x

    def augment(self, x, y):
        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))
        if hflip:
            x = TF.hflip(x)
            y = TF.hflip(y)
            # Adjust angles
            x[self.indices_of_degree_features, ...] = 360 - \
                x[self.indices_of_degree_features, ...]

        if vflip:
            x = TF.vflip(x)
            y = TF.vflip(y)
            # Adjust angles
            x[self.indices_of_degree_features, ...] = (
                180 - x[self.indices_of_degree_features, ...]) % 360

        if rotate != 0:
            angle = rotate * 90
            x = TF.rotate(x, angle)
            y = torch.unsqueeze(y, 0)
            y = TF.rotate(y, angle)
            y = torch.squeeze(y, 0)

            # Adjust angles
            x[self.indices_of_degree_features, ...] = (x[self.indices_of_degree_features,
                                                          ...] - 90 * rotate) % 360

        return x, y

        

    # define a function to load a batch of data from disk
    def load_data(self, indices):
        # load a chunk of data from disk
        data_chunk = np.load(self.image_path, mmap_mode='r')[indices]
        label_chunk = np.load(self.label_path, mmap_mode='r')[indices]

        #if self.n_channel==6:
        #    img_dataset = data_chunk[2:, :, :, :]
        #else:
        #    img_dataset = data_chunk[:, :, :, :]
        img_dataset = data_chunk[:]
        label_dataset = label_chunk[:]#[[self.label_sel], :, :, :]
        # 0 NIFC 1 VIIRS AF ACC 2 combine
        if self.target_is_single_day:
            y_dataset = np.zeros((2, 256, 256)) 
        else:   
            y_dataset = np.zeros((2, self.ts_length, 256, 256))
        # y_dataset = np.where(label_dataset > 0, 1, 0)
        # y_dataset = np.where(af_dataset > 0, 2, y_dataset)
        y_dataset[0, ...] = label_dataset[..., :] == 0
        y_dataset[1, ...] = label_dataset[..., :] > 0

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
    image_path = os.path.join(root_path, 'dataset_train/'+mode+'_train_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
    label_path = os.path.join(root_path, 'dataset_train/'+mode+'_train_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
    transform = Normalize(mean = [18.76488,27.441864,20.584806,305.99478,294.31738,14.625097,276.4207,275.16766],
                        std = [15.911591,14.879259,10.832616,21.761852,24.703484,9.878246,40.64329,40.7657])
    train_dataset = FireDataset(image_path=image_path, label_path=label_path, transform=transform, ts_length=ts_length, n_channel=8)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    print(next(iter(train_dataloader)).get('data').shape)