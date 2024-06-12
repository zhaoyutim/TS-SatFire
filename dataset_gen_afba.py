import argparse
import pandas as pd
import os
from satimg_dataset_processor.satimg_dataset_processor import AFBADatasetProcessor, AFTestDatasetProcessor
# Training set rois
dfs = []
for year in ['2017', '2018', '2019', '2020']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
# Test set rois
dfs_test = []
for year in ['2021']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df_test = pd.read_csv(filename)
    dfs_test.append(df_test)
df_test = pd.concat(dfs_test, ignore_index=True)

val_ids = ['20568194', '20701026','20562846','20700973','24462610', '24462788', '24462753', '24103571', '21998313', '21751303', '22141596', '21999381', '22712904']

df = df.sort_values(by=['Id'])
df['Id'] = df['Id'].astype(str)
train_df = df[~df.Id.isin(val_ids)]
val_df = df[df.Id.isin(val_ids)]

train_ids = train_df['Id'].values.astype(str)
val_ids = val_df['Id'].values.astype(str)

df_test = df_test.sort_values(by=['Id'])
test_ids = df_test['Id'].values.astype(str)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-mode', type=str, help='Train/Val/Test')
    parser.add_argument('-ts', type=int, help='Length of TS')
    parser.add_argument('-it', type=int, help='Interval')
    parser.add_argument('-uc', type=str, help='use case')
    args = parser.parse_args()
    ts_length = args.ts
    interval = args.it
    modes = args.mode
    usecase=args.uc
    if modes == 'train':
        locations = train_ids
    elif modes == 'val':
        locations = val_ids
    else:
        if usecase == 'ba':
            locations = test_ids
        else:
            locations = ['elephant_hill_fire', 'eagle_bluff_fire', 'double_creek_fire','sparks_lake_fire', 'lytton_fire', 
                        'chuckegg_creek_fire', 'swedish_fire', 'sydney_fire', 'thomas_fire', 'tubbs_fire', 
                        'carr_fire', 'camp_fire', 'creek_fire', 'blue_ridge_fire', 'dixie_fire', 'mosquito_fire', 'calfcanyon_fire']
    
    satimg_processor = AFBADatasetProcessor()
    if modes == 'train' or modes == 'val':
        satimg_processor.dataset_generator_seqtoseq(mode=modes, usecase=usecase, data_path='/home/z/h/zhao2/CalFireMonitoring/data/', locations=locations, visualize=False, 
                                                    file_name=usecase+'_'+modes+'_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                    label_name=usecase+'_'+modes+'_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                    save_path = 'dataset/dataset_'+modes, ts_length=ts_length, 
                                                    interval=interval, image_size=(256, 256))
    else:  
        for id in locations:
            if usecase == 'ba':
                satimg_processor.dataset_generator_seqtoseq(mode = 'test', usecase=usecase, data_path='/home/z/h/zhao2/CalFireMonitoring/data/', locations=[id], visualize=False, file_name=usecase+'_'+id+'_img_seqtoseql_'+str(ts_length)+'i_'+str(interval)+'.npy', label_name=usecase+'_'+id+'_label_seqtoseql_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                        save_path='dataset/dataset_test', ts_length=ts_length, interval=ts_length, rs_idx=0.3, cs_idx=0.3, image_size=(256, 256))
            else:
                af_test_processor = AFTestDatasetProcessor()
                af_test_processor.af_test_dataset_generator(id, save_path='dataset/dataset_test', file_name ='af_' + id + '_img.npy')
                af_test_processor.af_seq_tokenizing_and_test_slicing(location=id, modes=modes, ts_length=ts_length, interval=interval, usecase=usecase, root_path='/home/z/h/zhao2/TS-SatFire/dataset', save_path='/home/z/h/zhao2/TS-SatFire/dataset')