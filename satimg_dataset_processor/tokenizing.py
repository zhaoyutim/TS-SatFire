from utils import SatProcessingUtils
import numpy as np
import argparse
if __name__=='__main__':
    import os
    tokenize_processor = SatProcessingUtils()
    locations = ['elephant_hill_fire', 'eagle_bluff_fire', 'double_creek_fire', 'sparks_lake_fire', 'lytton_fire', 'chuckegg_creek_fire', 'swedish_fire',
                 'sydney_fire', 'thomas_fire', 'tubbs_fire', 'carr_fire', 'camp_fire',
                 'creek_fire', 'blue_ridge_fire', 'dixie_fire', 'mosquito_fire', 'calfcanyon_fire']
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-mode', type=str, help='Train/Val/Test')
    parser.add_argument('-ts', type=int, help='Length of TS')
    parser.add_argument('-it', type=int, help='Interval')
    parser.add_argument('-uc', type=str, help='use case')
    args = parser.parse_args()
    ts_length = args.ts
    interval = args.it
    uc = args.uc
    modes = args.mode
    if modes == 'train':
        locations = ['train', 'val']
    else:
        locations = locations
    window_size = 1
    for location in locations:
        if location in ['val', 'train']:
            root_path = f'/home/z/h/zhao2/TS-SatFire/dataset/dataset_{location}'
            save_path = f'/home/z/h/zhao2/TS-SatFire/dataset/dataset_{location}'
            tokenized_array = np.load(os.path.join(root_path, f'af_{location}_img_seqtoseq_alll_{ts_length}i_{interval}.npy')).transpose((0, 3, 4, 2, 1))
            tokenized_label = np.load(os.path.join(root_path, f'af_{location}_label_seqtoseq_alll_{ts_length}i_{interval}.npy')).transpose((0, 3, 4, 2, 1))
            tokenized_label = tokenized_label[..., 2]
        else:
            root_path = '/home/z/h/zhao2/CalFireMonitoring/data_train_proj2'
            save_path = '/home/z/h/zhao2/TS-SatFire/dataset/dataset_test'
            tokenized_array = np.load(os.path.join(root_path, f'af_{location}_img.npy')).transpose((0, 3, 4, 1, 2))
            tokenized_label = np.load(os.path.join(root_path, f'af_{location}_label.npy')).transpose((0, 2, 3, 1))
        if tokenized_array.shape[-2]>=ts_length:
            lb = (tokenized_array.shape[-2]-ts_length)//2
            rb = (tokenized_array.shape[-2]+ts_length)//2
            tokenized_array = tokenized_array[:,:,:,lb:rb,:]
            tokenized_label = tokenized_label[:,:,:,lb:rb]
        if uc == 'temp':
            print('tokenizing')
            tokenized_array = np.nan_to_num(tokenized_array).reshape(-1,tokenized_array.shape[-2],tokenized_array.shape[-1])
            tokenized_label = np.nan_to_num(tokenized_label).reshape(-1,tokenized_label.shape[-1])
            print(tokenized_array.shape)
            print(tokenized_label.shape)
            np.save(os.path.join(save_path,f'af_{location}_img_seqtoseq_l{ts_length}_w{window_size}.npy'), tokenized_array)
            np.save(os.path.join(save_path,f'af_{location}_label_seqtoseq_l{ts_length}_w{window_size}.npy'), tokenized_label)
        else:
            img_array = np.nan_to_num(np.squeeze(tokenized_array).transpose((2,3,0,1)))
            img_label = np.nan_to_num(np.squeeze(tokenized_label).transpose((2,0,1)))
            img_label = np.repeat(img_label[:,np.newaxis,:,:], 3, axis=1)
            img_array = img_array[:,:,np.newaxis,:,:]
            img_label = img_label[:,:,np.newaxis,:,:]
            print(img_array.shape)
            print(img_label.shape)
            np.save(os.path.join(save_path,f'af_{location}_img_seqtoseql_{1}i_{1}.npy'), img_array.astype(np.float32))
            np.save(os.path.join(save_path,f'af_{location}_label_seqtoseql_{1}i_{1}.npy'), img_label.astype(np.float32))