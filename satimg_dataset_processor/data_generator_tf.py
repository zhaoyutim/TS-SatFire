import os
import numpy as np
import tensorflow as tf

class FireDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, mode, train_test, ts_length=6, interval=3, batch_size=512, input_shape=(6, 8), n_channels=8,
                 n_classes=2, shuffle=True):
        # 'Initialization"
        root_path = '/home/z/h/zhao2/TS-SatFire/dataset/'
        if train_test=='train':
            img_path = os.path.join(root_path, 'dataset_train/dataset_train'+mode+'_train_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
            label_path = os.path.join(root_path, 'dataset_train/dataset_train'+mode+'_train_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
        elif train_test=='val':
            img_path = os.path.join(root_path, 'dataset_val/'+mode+'_val_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
            label_path = os.path.join(root_path, 'dataset_val/'+mode+'_val_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
        else:
            print('not implemented')
    
        self.img = np.load(img_path, mmap_mode='r').transpose((0, 3, 4, 2, 1))
        if mode=='af':
            self.img = self.img.reshape(self.img.shape[0]* self.img.shape[1]*self.img.shape[2], self.img.shape[3], self.img.shape[4])
        
        self.label_all = np.load(label_path, mmap_mode='r').transpose((0, 3, 4, 2, 1))
        if mode=='af':
            self.label_all = self.label_all.reshape(self.label_all.shape[0]* self.label_all.shape[1]*self.label_all.shape[2], self.label_all.shape[3], self.label_all.shape[4])
        self.label = np.zeros((self.label_all.shape[0],self.label_all.shape[1], self.label_all.shape[2], self.label_all.shape[3], 2))
        self.label[..., 0] = self.label_all[..., 2] == 0
        self.label[..., 1] = self.label_all[..., 2] > 0

        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(self.img.shape[0] / self.batch_size))
    
    def __normalize(self, array):
        mean = [18.76488,27.441864,20.584806,305.99478,294.31738,14.625097,276.4207,275.16766]
        std = [15.911591,14.879259,10.832616,21.761852,24.703484,9.878246,40.64329,40.7657]
        for i in range(self.n_channels):
            array[..., i] = (array[..., i]-mean[i])/std[i]
        return array
    
    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.img[indexes], self.label[indexes]

        return self.__normalize(X), y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(self.img.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

if __name__=='__main__':
    data_gen = FireDataGenerator('af', train_test='train', ts_length=6, interval=3, batch_size=4, input_shape=(6,8), n_channels=8, n_classes=2)
    x_batch_train = data_gen[0][0]
    b, h, w, t, c = x_batch_train.shape[0], x_batch_train.shape[1], x_batch_train.shape[2], x_batch_train.shape[3], x_batch_train.shape[4]
    x_batch_train = tf.reshape(x_batch_train, (b*h*w, t, c))
    print(x_batch_train.shape)
    print(len(data_gen))
