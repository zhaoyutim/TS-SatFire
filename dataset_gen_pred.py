import pandas as pd
import argparse

from satimg_dataset_processor.satimg_dataset_processor import AFBADatasetProcessor, PredDatasetProcessor
dfs = []
for year in ['2017', '2018', '2019', '2020']:
# for year in ['2021']:
    filename = '/home/z/h/zhao2/CalFireMonitoring/roi/us_fire_' + year + '_out_new.csv'
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
dfs_test = []
for year in ['2021']:
    filename = '/home/z/h/zhao2/CalFireMonitoring/roi/us_fire_' + year + '_out_new.csv'
    df_test = pd.read_csv(filename)
    dfs_test.append(df_test)
df_test = pd.concat(dfs_test, ignore_index=True)
val_ids = ['24462610', '24462788', '24462753']
test_ids = ['24461623', '24332628']
skip_ids = ['21890069', '20777160', '20777163', '20777166']
target_ids = ['21889672', '21889683', '21889697', '21889719', '21889734', '21889754', '21997775']

df = df.sort_values(by=['Id'])
df['Id'] = df['Id'].astype(str)
train_df = df[~df.Id.isin(val_ids + skip_ids + test_ids)]
val_df = df[df.Id.isin(val_ids)]
test_df = df[df.Id.isin(test_ids)]
target_df = df[df.Id.isin(target_ids)]

train_ids = train_df['Id'].values.astype(str)
val_ids = val_df['Id'].values.astype(str)

df_test = df_test.sort_values(by=['Id'])
test_ids = df_test['Id'].values.astype(str)
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
        satimg_processor.pred_dataset_generator_seqtoseq(mode=modes, locations=locations, visualize=True, 
                                                file_name=usecase+'_'+modes+'_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                label_name=usecase+'_'+modes+'_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                save_path = 'dataset/dataset_'+modes, ts_length=ts_length, 
                                                interval=interval, image_size=(256, 256))
    # ids = ['donnie_creek', 'slave_lake']
    else:
        import os
        for id in locations:
            print(id)
            os.system(f'gdrive upload -p 1EagJKmme73iHvJT9e03kS8GgBHOWkqnM dataset/dataset_test/pred_{id}_img_seqtoseql_6i_3.npy')
            os.system(f'gdrive upload -p 1EagJKmme73iHvJT9e03kS8GgBHOWkqnM dataset/dataset_test/pred_{id}_label_seqtoseql_6i_3.npy')
            # satimg_processor.pred_dataset_generator_seqtoseq(mode = 'test', locations=[id], visualize=True, file_name=usecase+'_'+id+'_img_seqtoseql_'+str(ts_length)+'i_'+str(interval)+'.npy', label_name=usecase+'_'+id+'_label_seqtoseql_'+str(ts_length)+'i_'+str(interval)+'.npy',
                                                        #    save_path='dataset/dataset_test', ts_length=ts_length, interval=ts_length, rs_idx=0.3, cs_idx=0.3, image_size=(256, 256))