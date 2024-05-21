from utils import SatProcessingUtils
import numpy as np

if __name__=='__main__':
    import os
    tokenize_processor = SatProcessingUtils()
    locations = ['elephant_hill_fire', 'eagle_bluff_fire', 'double_creek_fire']
    locations += ['sparks_lake_fire', 'lytton_fire', 'chuckegg_creek_fire', 'swedish_fire',
                 'sydney_fire', 'thomas_fire', 'tubbs_fire', 'carr_fire', 'camp_fire',
                 'creek_fire', 'blue_ridge_fire', 'dixie_fire', 'mosquito_fire', 'calfcanyon_fire']
    root_path = '/home/z/h/zhao2/CalFireMonitoring/data_train_proj2'
    save_path = '/home/z/h/zhao2/TS-SatFire/dataset/dataset_train'
    window_size = 1
    ts_length=6
    for location in locations:
        tokenized_array = tokenize_processor.tokenizing(os.path.join(root_path, f'proj3_{location}_img.npy'), window_size)
        np.nan_to_num(tokenized_array)
        np.save(os.path.join(save_path,f'af_{location}_img_seqtoone_l{ts_length}_w{window_size}.npy'), tokenized_array.reshape(-1,tokenized_array.shape[-2],tokenized_array.shape[-1]))