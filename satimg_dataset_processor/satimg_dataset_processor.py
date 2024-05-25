import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from satimg_dataset_processor.utils import SatProcessingUtils

class AFBADatasetProcessor(SatProcessingUtils):
    def dataset_generator_seqtoseq(self, mode, usecase, locations, file_name, label_name, save_path, rs_idx=0, cs_idx=0,
                                               visualize=True, ts_length=10, interval=3, image_size=(224, 224)):
        satellite_day = 'VIIRS_Day'
        stack_over_location = []
        stack_label_over_locations = []
        n_channels = 8
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_path = '/home/z/h/zhao2/CalFireMonitoring/data/' + location + '/' + satellite_day + '/'
            file_list = glob(data_path + '/*.tif')
            file_list.sort()
            if len(file_list) == 0:
                print('empty file list')
                continue
            preprocessing = SatProcessingUtils()
            array_day, _ = preprocessing.read_tiff(file_list[0])
            array_stack = []
            label_stack = []

            # if mode == 'train' or mode == 'val':
            output_shape_x = 256
            output_shape_y = 256
            offset=128
            # else:
            #     output_shape_x = array_day.shape[1]
            #     output_shape_y = array_day.shape[2]
            #     offset=0
            
            original_shape_x = array_day.shape[1]
            original_shape_y = array_day.shape[2]

            ba_label = np.zeros((output_shape_x, output_shape_y))
            af_acc_label = np.zeros((output_shape_x, output_shape_y))
            new_base_acc_label = af_acc_label
            new_base_ba_label = ba_label
            file_list_size = len(file_list)
            max_img = np.zeros((n_channels, output_shape_x, output_shape_y), dtype=np.float32)
            for i in range(0, file_list_size, interval):
                if i + ts_length >= file_list_size:
                    print('drop the tail')
                    break
                output_array = np.zeros((ts_length, n_channels, output_shape_x, output_shape_y), dtype=np.float32)
                output_label = np.zeros((ts_length, 3, output_shape_x, output_shape_y), dtype=np.float32)
                for j in range(ts_length):
                    file = file_list[j + i]
                    array_day, _ = preprocessing.read_tiff(file)
                    if os.path.exists(file.replace('VIIRS_Day', 'VIIRS_Night')):
                        array_night, _ = preprocessing.read_tiff(file.replace('VIIRS_Day', 'VIIRS_Night'))
                        if array_night.shape[0] == 5:
                            print('Day_night miss align')
                            array_night = array_night[3:, :, :]
                        if array_night.shape[0] < 2:
                            print(file.replace('VIIRS_Day', 'VIIRS_Night'), 'band incomplete')
                            continue
                        if array_night.shape[1] != array_day.shape[1] or array_night.shape[2] != array_day.shape[2]:
                            print('Day Night not match')
                            print(file)
                    else:
                        array_night = np.zeros((2, original_shape_x, original_shape_y))
                    print(file)
                    img = np.concatenate((array_day[:6, offset:output_shape_x+offset, offset:output_shape_y+offset], array_night[:, offset:output_shape_x+offset, offset:output_shape_y+offset]), axis=0)
                    img = np.nan_to_num(img[:,:output_shape_x, :output_shape_y])
                    max_img = np.maximum(img, max_img)
                    if usecase=='ba':
                        print('usecase ba')
                        img = np.concatenate((img[:3,...],max_img[3:5,...],img[[5],...],max_img[6:8,...]))
                    elif usecase=='af':
                        print('usecase af')
                        img = np.concatenate((img[:3,...],img[3:5,...],img[[5],...],img[6:8,...]))
                    else:
                        raise "no support usecase"
                    ba_img = np.concatenate(([img[[5],:,:], img[[1],:,:], img[[0],:,:]]))
                    if array_day.shape[0]==8:
                        label = np.nan_to_num(array_day[7, :, :])
                    else:
                        label = np.zeros((output_shape_x, output_shape_y))
                    af= array_day[6, :, :]

                    ba_img = ba_img/40
                    label = np.nan_to_num(label[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    af = np.nan_to_num(af[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    ba_label = np.logical_or(label, ba_label)
                    af_acc_label = np.logical_or(af, af_acc_label)
                    if j == interval-1:
                        new_base_acc_label = af_acc_label
                        new_base_ba_label = ba_label
                    output_array[j, :n_channels, :, :] = img
                    output_label[j, 0, :, :] = np.logical_or(af_acc_label, ba_label)
                    output_label[j, 1, :, :] = af_acc_label
                    output_label[j, 2, :, :] = af
                    if visualize:
                        plt.figure(figsize=(12, 4), dpi=80)
                        plt.subplot(131)
                        plt.imshow(ba_img.transpose((1,2,0)))
                        plt.imshow(np.where(af_acc_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1, vmin=0, vmax=1)
                        plt.axis('off')
                        plt.title('AF ACC')
                        plt.subplot(132)
                        plt.imshow(ba_img.transpose((1,2,0)))
                        plt.imshow(np.where(af==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1, vmin=0, vmax=1)
                        plt.axis('off')
                        plt.title('AF')
                        plt.subplot(133)
                        plt.imshow(ba_img.transpose((1,2,0)))
                        plt.imshow(np.where(ba_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1, vmin=0, vmax=1)
                        plt.axis('off')
                        plt.title('BA')
                        plt.savefig(save_path+'_figure/'+location+'_sequence_'+str(i)+'_time_'+str(j)+'_ts_'+str(ts_length)+'_comb.png', bbox_inches='tight')
                af_acc_label = new_base_acc_label
                ba_label = new_base_ba_label
                array_stack.append(output_array)
                label_stack.append(output_label)
            if len(array_stack)==0:
                print('No enough TS')
                continue
            output_array_stacked = np.stack(array_stack, axis=0)
            output_label_stacked = np.stack(label_stack, axis=0)
            stack_over_location.append(output_array_stacked)
            stack_label_over_locations.append(output_label_stacked)
        dataset_stacked_over_locations = np.concatenate(stack_over_location, axis=0).transpose((0,2,1,3,4))
        labels_stacked_over_locations = np.concatenate(stack_label_over_locations, axis=0).transpose((0,2,1,3,4))
        del stack_over_location
        del stack_label_over_locations
        for i in range(8):
            print(np.nanmean(dataset_stacked_over_locations[:,i,:,:,:]))
            print(np.nanstd(dataset_stacked_over_locations[:,i,:,:,:]))
        np.save(save_path + '/' + file_name, dataset_stacked_over_locations.astype(np.float32))
        np.save(save_path + '/' + label_name, labels_stacked_over_locations.astype(np.float32))


class PredDatasetProcessor(SatProcessingUtils):
    def pred_dataset_generator_seqtoseq(self, mode, locations, file_name, label_name, save_path, rs_idx=0, cs_idx=0,
                                               visualize=True, ts_length=10, interval=3, image_size=(224, 224)):
        satellite_day = 'VIIRS_Day'
        stack_over_location = []
        stack_label_over_locations = []
        n_channels = 8+19
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_path = '/home/z/h/zhao2/CalFireMonitoring/data/' + location + '/' + satellite_day + '/'
            file_list = glob(data_path + '/*.tif')
            file_list.sort()
            if len(file_list) == 0:
                print('empty file list')
                continue
            preprocessing = SatProcessingUtils()
            array_day, _ = preprocessing.read_tiff(file_list[0])
            array_stack = []
            label_stack = []

            output_shape_x = 256
            output_shape_y = 256
            offset=128
            
            original_shape_x = array_day.shape[1]
            original_shape_y = array_day.shape[2]

            ba_label = np.zeros((output_shape_x, output_shape_y))
            af_acc_label = np.zeros((output_shape_x, output_shape_y))
            new_base_acc_label = af_acc_label
            new_base_ba_label = ba_label
            max_img = np.zeros((n_channels, output_shape_x, output_shape_y), dtype=np.float32)
            file_list_size = len(file_list)
            for i in range(0, file_list_size, interval):
                if i + ts_length >= file_list_size:
                    print('drop the tail')
                    break
                output_array = np.zeros((ts_length, n_channels, output_shape_x, output_shape_y), dtype=np.float32)
                output_label = np.zeros((output_shape_x, output_shape_y), dtype=np.float32)
                for j in range(ts_length+1):
                    file = file_list[j + i]
                    array_day, _ = preprocessing.read_tiff(file)
                    if os.path.exists(file.replace('VIIRS_Day', 'VIIRS_Night')):
                        array_night, _ = preprocessing.read_tiff(file.replace('VIIRS_Day', 'VIIRS_Night'))
                        if array_night.shape[0] == 5:
                            print('Day_night miss align')
                            array_night = array_night[3:, :, :]
                        if array_night.shape[0] < 2:
                            print(file.replace('VIIRS_Day', 'VIIRS_Night'), 'band incomplete')
                            continue
                        if array_night.shape[1] != array_day.shape[1] or array_night.shape[2] != array_day.shape[2]:
                            print('Day Night not match')
                            print(file)
                    else:
                        array_night = np.zeros((2, original_shape_x, original_shape_y))
                    array_pred, _ = preprocessing.read_tiff(file.replace('VIIRS_Day', 'FirePred'))
                    print(file)
                    img = np.concatenate((array_day[:6, offset:output_shape_x+offset, offset:output_shape_y+offset], array_night[:, offset:output_shape_x+offset, offset:output_shape_y+offset], array_pred[:, offset:output_shape_x+offset, offset:output_shape_y+offset]), axis=0)
                    img = np.nan_to_num(img[:,:output_shape_x, :output_shape_y])
                    max_img = np.maximum(img, max_img)
                    # if use max image
                    img = np.concatenate((img[:3,...],max_img[3:5,...],img[[5],...],max_img[6:8,...],img[8:,...]))
                    print(img.shape)
                    ba_img = img[3,:,:]
                    if array_day.shape[0]==8:
                        label = np.nan_to_num(array_day[7, :, :])
                    else:
                        label = np.zeros((output_shape_x, output_shape_y))
                    af= array_day[6, :, :]

                    ba_img = (ba_img-ba_img.min())/(ba_img.max()-ba_img.min())
                    label = np.nan_to_num(label[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    af = np.nan_to_num(af[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    ba_label = np.logical_or(label, ba_label)
                    af_acc_label = np.logical_or(af, af_acc_label)
                    if j == interval-1:
                        new_base_acc_label = af_acc_label
                        new_base_ba_label = ba_label
                    if j <ts_length:
                        output_array[j, :n_channels, :, :] = img
                    if j == ts_length:
                        output_label[:, :] = af_acc_label
                    if visualize and j == ts_length:
                        plt.figure(figsize=(8, 4), dpi=80)
                        plt.subplot(121)
                        plt.imshow(ba_img)
                        plt.axis('off')
                        plt.title('Band I4 Day')
                        plt.subplot(122)
                        plt.imshow(ba_img)
                        plt.imshow(np.where(output_label[:, :]==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        plt.axis('off')
                        plt.title('BA next day')
                        plt.savefig(save_path+'_figure/'+location+'_sequence_'+str(i)+'_time_'+str(j)+'_ts_'+str(ts_length)+'_comb_pred.png', bbox_inches='tight')
                af_acc_label = new_base_acc_label
                ba_label = new_base_ba_label
                array_stack.append(output_array)
                label_stack.append(output_label)
            if len(array_stack)==0:
                print('No enough TS')
                continue
            output_array_stacked = np.stack(array_stack, axis=0)
            output_label_stacked = np.stack(label_stack, axis=0)
            stack_over_location.append(output_array_stacked)
            stack_label_over_locations.append(output_label_stacked)
        dataset_stacked_over_locations = np.concatenate(stack_over_location, axis=0).transpose((0,2,1,3,4))
        labels_stacked_over_locations = np.concatenate(stack_label_over_locations, axis=0)
        del stack_over_location
        del stack_label_over_locations
        for i in range(n_channels):
            print(np.nanmean(dataset_stacked_over_locations[:,i,:,:,:]))
            print(np.nanstd(dataset_stacked_over_locations[:,i,:,:,:]))
        np.save(save_path + '/' + file_name, dataset_stacked_over_locations.astype(np.float32))
        np.save(save_path + '/' + label_name, labels_stacked_over_locations.astype(np.float32))