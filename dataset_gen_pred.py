import pandas as pd
import argparse
import os
from satimg_dataset_processor.satimg_dataset_processor import PredDatasetProcessor
dfs = []
for year in ['2017', '2018', '2019', '2020']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
dfs_test = []
for year in ['2021']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df_test = pd.read_csv(filename)
    dfs_test.append(df_test)
df_test = pd.concat(dfs_test, ignore_index=True)
val_ids = ['20568194', '20701026','20562846','20700973','24462610', '24462788', '24462753', '24103571', '21998313', '21751303', '22141596', '21999381', '23301962', '22712904', '22713339']

df = df.sort_values(by=['Id'])
df['Id'] = df['Id'].astype(str)
train_df = df[~df.Id.isin(val_ids)]
val_df = df[df.Id.isin(val_ids)]

train_ids = train_df['Id'].values.astype(str)
val_ids = val_df['Id'].values.astype(str)

df_test = df_test.sort_values(by=['Id'])
test_ids = df_test['Id'].values.astype(str)
test_label_sel = df_test['label_sel'].values.astype(int)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-mode', type=str, help='Train/Val/Test')
    parser.add_argument('-ts', type=int, help='Length of TS')
    parser.add_argument('-it', type=int, help='Interval')
    args = parser.parse_args()
    ts_length = args.ts
    interval = args.it
    modes = args.mode
    if modes == 'train':
        locations = train_ids
    elif modes == 'val':
        locations = val_ids
    else:
        locations = test_ids
    usecase='pred'
    satimg_processor = PredDatasetProcessor()
    if modes in ['train', 'val']:
        satimg_processor.pred_dataset_generator_seqtoseq(mode=modes, locations=locations, visualize=False, data_path='/home/z/h/zhao2/CalFireMonitoring/data/',
                                                file_name=usecase+'_'+modes+'_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                label_name=usecase+'_'+modes+'_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                save_path = 'dataset/dataset_'+modes, ts_length=ts_length, 
                                                interval=interval, image_size=(256, 256))
    else:
        for i, id in enumerate(locations):
            print(id)
            satimg_processor.pred_dataset_generator_seqtoseq(mode = 'test', locations=[id], visualize=False, data_path='/home/z/h/zhao2/CalFireMonitoring/data/',file_name=usecase+'_'+id+'_img_seqtoseql_'+str(ts_length)+'i_'+str(interval)+'.npy', label_name=usecase+'_'+id+'_label_seqtoseql_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                           save_path='dataset/dataset_test', ts_length=ts_length, interval=interval, rs_idx=0.3, cs_idx=0.3, image_size=(256, 256), label_sel=test_label_sel[i])