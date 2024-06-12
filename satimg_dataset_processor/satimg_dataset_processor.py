import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from satimg_dataset_processor.utils import SatProcessingUtils

class AFBADatasetProcessor(SatProcessingUtils):
    def dataset_generator_seqtoseq(self, mode, usecase, data_path, locations, file_name, label_name, save_path, rs_idx=0, cs_idx=0,
                                               visualize=True, ts_length=10, interval=3, image_size=(224, 224)):
        satellite_day = 'VIIRS_Day'
        stack_over_location = []
        stack_label_over_locations = []
        n_channels = 8
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            study_area_path = data_path + '/' + location + '/' + satellite_day + '/'
            file_list = glob(study_area_path + '/*.tif')
            file_list.sort()
            if len(file_list) == 0:
                print('empty file list')
                continue
            array_day, _ = self.read_tiff(file_list[0])
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
            file_list_size = len(file_list)
            max_img = np.zeros((n_channels, output_shape_x, output_shape_y), dtype=np.float32)
            for i in range(0, file_list_size, interval):
                if i + ts_length >= file_list_size:
                    i=file_list_size-ts_length
                    print('append the tail')
                    # break
                output_array = np.zeros((ts_length, n_channels, output_shape_x, output_shape_y), dtype=np.float32)
                output_label = np.zeros((ts_length, 3, output_shape_x, output_shape_y), dtype=np.float32)
                for j in range(ts_length):
                    file = file_list[j + i]
                    array_day, _ = self.read_tiff(file)
                    if os.path.exists(file.replace('VIIRS_Day', 'VIIRS_Night')):
                        array_night, _ = self.read_tiff(file.replace('VIIRS_Day', 'VIIRS_Night'))
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
    def pred_dataset_generator_seqtoseq(self, mode, locations, data_path, file_name, label_name, save_path, rs_idx=0, cs_idx=0,
                                               visualize=True, ts_length=10, interval=3, image_size=(224, 224), label_sel=1):
        satellite_day = 'VIIRS_Day'
        stack_over_location = []
        stack_label_over_locations = []
        n_channels = 8+19
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_day_path = data_path + location + '/' + satellite_day + '/'
            file_list = glob(data_day_path + '/*.tif')
            file_list.sort()
            if len(file_list) == 0:
                print('empty file list')
                continue
            array_day, _ = self.read_tiff(file_list[0])
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
                    array_day, _ = self.read_tiff(file)
                    if os.path.exists(file.replace('VIIRS_Day', 'VIIRS_Night')):
                        array_night, _ = self.read_tiff(file.replace('VIIRS_Day', 'VIIRS_Night'))
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
                    array_pred, _ = self.read_tiff(file.replace('VIIRS_Day', 'FirePred'))
                    img = np.concatenate((array_day[:6, offset:output_shape_x+offset, offset:output_shape_y+offset], array_night[:, offset:output_shape_x+offset, offset:output_shape_y+offset], array_pred[:, offset:output_shape_x+offset, offset:output_shape_y+offset]), axis=0)
                    img = (img[:,:output_shape_x, :output_shape_y])
                    max_img = np.maximum(img, max_img)
                    img = np.concatenate((img[:3,...],max_img[3:5,...],img[[5],...],max_img[6:8,...],img[8:,...]))
                    ba_img = img[3,:,:]
                    if array_day.shape[0]==8:
                        label = (array_day[7, :, :])
                    else:
                        label = np.zeros((output_shape_x, output_shape_y))
                    af= array_day[6, :, :]

                    ba_img = (ba_img-ba_img.min())/(ba_img.max()-ba_img.min())
                    label = (label[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    af = (af[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    ba_label = np.logical_or(label, ba_label)
                    af_acc_label = np.logical_or(af, af_acc_label)
                    if label_sel==1:
                        final_label = af_acc_label
                    else:
                        final_label = np.logical_or(af_acc_label, ba_label)
                    
                    if j == interval-1:
                        new_base_acc_label = af_acc_label
                        new_base_ba_label = ba_label
                    if j <ts_length:
                        prev_final_label = final_label.copy()
                        output_array[j, :n_channels, :, :] = img
                    if j == ts_length:
                        output_label[:, :] = np.where(np.logical_and(prev_final_label==0, final_label>0), 1, 0)
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

class AFTestDatasetProcessor(SatProcessingUtils):  
    def af_test_dataset_generator(self, location, file_name, save_path, image_size=(256, 256)):
        satellite = 'VIIRS_Day'
        window_size = 3
        ts_length = 10
        n_channels = 8
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        print(location)
        data_path = 'data/' + location + '/' + satellite + '/'
        file_list = glob(data_path + '/*.tif')
        file_list.sort()
        if len(file_list) % ts_length != 0:
            num_sequence = len(file_list) // ts_length + 1
        else:
            num_sequence = len(file_list) // ts_length
        array_day, _ = self.read_tiff(file_list[0])

        padding = window_size // 2
        array_stack = []
        th = {
            'elephant_hill_fire': [(340, 330), (340, 337), (330, 330), (335, 330), (325, 330), (330, 330), (330, 330),
        (330, 330), (340, 330), (335, 330)],
            'eagle_bluff_fire': [(335, 330), (335, 337), (333, 335), (335, 330), (335, 330), (330, 330), (330, 330),
                                 (335, 330), (337, 330), (340, 330)],
            'double_creek_fire': [(360, 330), (340, 337), (340, 325), (335, 330), (337, 330), (337, 330), (335, 335),
                                  (335, 330), (333, 330), (335, 330)],
            'sparks_lake_fire': [(360, 330), (340, 337), (340, 325), (335, 330), (337, 330), (337, 330), (335, 335),
                                  (335, 330), (340, 330), (340, 330)],
            'lytton_fire': [(357, 330), (340, 337), (340, 325), (335, 330), (337, 330), (340, 330), (360, 335),
                                  (340, 330), (345, 330), (340, 330)],
            'chuckegg_creek_fire': [(340, 330), (320, 337), (325, 330), (330, 330), (330, 330), (340, 330), (330, 330),
                                    (325, 330), (325, 330), (338, 330)],
            'swedish_fire': [(340, 330), (320, 337), (320, 330), (330, 330), (330, 330), (330, 330), (330, 330),
                                    (325, 330), (325, 330), (330, 330)],
            'sydney_fire': [(325, 330), (330, 337), (335, 330), (334, 330), (330, 330), (325, 340), (345, 330),
                            (340, 330), (330, 330), (330, 330)],
            'thomas_fire': [(335, 330), (330, 337), (320, 335), (325, 330), (330, 330), (330, 330), (325, 330),
                            (330, 330), (330, 330), (340, 330)],
            'tubbs_fire': [(340, 330), (325, 337), (330, 330), (320, 330), (325, 330), (325, 330), (330, 330),
                           (330, 330), (350, 350), (320, 330)],
            'carr_fire': [(333, 330), (339, 337), (343, 335), (343, 330), (337, 330), (335, 330), (330, 330),
                          (335, 330), (337, 330), (340, 330)],
            'camp_fire': [(335, 330), (320, 337), (320, 310), (308, 330), (310, 330), (305, 330), (320, 330),
                          (315, 330), (310, 330), (310, 330)],
            'kincade_fire': [(330, 330), (320, 337), (335, 335), (330, 330), (330, 330), (330, 330), (320, 330),
                             (330, 330), (350, 330), (340, 330)],
            'creek_fire': [(355, 330), (340, 337), (335, 335), (330, 330), (330, 330), (330, 330), (335, 330),
                           (340, 330), (337, 330), (340, 330)],
            'blue_ridge_fire': [(330, 330), (325, 337), (350, 335), (340, 330), (335, 330), (330, 330), (330, 330),
                                (335, 330), (337, 330), (340, 330)],
            'dixie_fire': [(340, 330), (335, 337), (345, 345), (340, 330), (345, 360), (340, 330), (333, 330),
                              (335, 330), (340, 350), (345, 350)],
            'mosquito_fire': [(335, 330), (335, 337), (335, 325), (340, 330), (340, 330), (335, 330), (330, 330),
                              (325, 330), (330, 330), (335, 330)],
            'calfcanyon_fire' :[(330, 330), (330, 337), (330, 325), (340, 330), (330, 330), (330, 330), (330, 330), (330, 330), (330, 330), (329, 330)]
        }

        th_night = {
            'elephant_hill_fire': [(300, 300), (310, 305), (310, 305), (315, 305), (305, 305), (305, 305), (315, 305),
         (305, 305), (305, 305), (315, 305)],
            'eagle_bluff_fire': [(300, 300), (310, 305), (310, 305), (310, 305), (298, 305), (305, 305), (315, 305),
         (305, 305), (305, 305), (315, 305)],
            'double_creek_fire': [(310, 300), (305, 305), (310, 305), (310, 305), (298, 305), (305, 305), (308, 305),
         (305, 305), (305, 305), (315, 305)],
            'sparks_lake_fire': [(310, 300), (310, 305), (305, 305), (315, 305), (305, 305), (308, 305), (310, 305),
         (310, 305), (305, 305), (315, 305)],
            'lytton_fire': [(320, 320), (310, 305), (320, 305), (315, 305), (305, 305), (308, 305), (300, 305),
         (300, 305), (305, 305), (304, 305)],
            'chuckegg_creek_fire': [(320, 320), (310, 305), (315, 305), (315, 305), (305, 305), (308, 305), (306, 305),
         (300, 305), (300, 305), (295, 305)],
            'swedish_fire': [(320, 320), (310, 305), (315, 305), (315, 305), (305, 305), (308, 305), (306, 305),
         (300, 305), (300, 305), (295, 305)],
            'sydney_fire': [(305, 320), (300, 305), (315, 305), (305, 305), (295, 305), (300, 305), (306, 305),
         (300, 305), (300, 305), (295, 305)],
            'thomas_fire': [(305, 320), (310, 305), (315, 305), (305, 305), (295, 305), (300, 305), (310, 305),
         (300, 305), (300, 305), (295, 305)],
            'tubbs_fire': [(305, 320), (310, 305), (315, 305), (305, 305), (295, 305), (300, 305), (315, 305),
         (300, 305), (300, 305), (300, 305)],
            'carr_fire': [(305, 320), (305, 305), (315, 305), (305, 305), (310, 305), (320, 305), (305, 305),
         (300, 305), (305, 305), (305, 305)],
            'camp_fire': [(305, 320), (305, 305), (315, 305), (305, 305), (310, 305), (320, 305), (305, 305),
         (300, 305), (305, 305), (295, 305)],
            'kincade_fire': [(305, 320), (310, 305), (315, 305), (305, 305), (310, 305), (310, 305), (305, 305),
         (300, 305), (305, 305), (315, 305)],
            'creek_fire': [(305, 320), (320, 305), (320, 305), (315, 305), (310, 305), (310, 305), (305, 305),
         (300, 305), (305, 305), (315, 305)],
            'blue_ridge_fire': [(305, 320), (320, 305), (320, 305), (315, 305), (310, 305), (310, 305), (305, 305),
         (300, 305), (310, 305), (315, 305)],
            'dixie_fire': [(300, 320), (320, 305), (320, 305), (315, 305), (310, 305), (310, 305), (305, 305),
         (300, 305), (310, 305), (315, 305)],
            'mosquito_fire': [(300, 320), (320, 305), (320, 305), (315, 305), (310, 305), (310, 305), (305, 305),
         (300, 305), (310, 305), (315, 305)],
            'calfcanyon_fire': [(300, 320), (320, 305), (320, 305), (300, 300), (305, 305), (300, 305), (305, 305),
         (300, 305), (310, 305), (310, 305)]
        }
        for j in range(num_sequence):
            output_array = np.zeros((ts_length, n_channels + 1, image_size[0], image_size[1]))
            if j == num_sequence - 1 and j != 0:
                file_list_size = len(file_list) % ts_length
            else:
                file_list_size = ts_length
            for i in range(file_list_size):
                file = file_list[i + j * 10]
                array_day, profile = self.read_tiff(file)
                array_night, _ = self.read_tiff(file.replace('VIIRS_Day', 'VIIRS_Night'))
                if os.path.exists(file.replace('VIIRS_Day', 'VIIRS_Night')):
                    array_night, _ = self.read_tiff(file.replace('VIIRS_Day', 'VIIRS_Night'))
                else:
                    array_night = np.zeros((2, array_day.shape[1], array_day.shape[2]))
                if array_day.shape[0] != 8:
                    print(file, 'band incomplete')
                    continue
                if array_night.shape[0] == 5:
                    print('Day_night miss align')
                    array_night = array_night[3:, :, :]
                if array_night.shape[0] < 2:
                    print(file.replace('VIIRS_Day', 'VIIRS_Night'), 'band incomplete')
                    continue
                if array_night.shape[1] != array_day.shape[1] or array_night.shape[2] != array_day.shape[2]:
                    print('Day Night not match')
                    print(file)
                    continue

                th_i= th[location]
                th_n = th_night[location]
                if not os.path.exists(f'{save_path}_figure'):
                    os.mkdir(f'{save_path}_figure')
                af = np.zeros(array_day[3, :, :].shape)
                af[:, :] = np.logical_or(array_day[3, :, :] > th_i[i][0], array_day[4, :, :] > th_i[i][1])
                af_img = af
                af_img[np.logical_not(af_img[:, :])] = np.nan
                plt.subplot(221)
                plt.title('day af')
                plt.imshow(array_day[3, :, :])
                plt.imshow(af_img, cmap='hsv', interpolation='nearest')
                plt.subplot(222)
                plt.title('original')
                plt.imshow(array_day[3, :, :])

                af_night = np.zeros(array_night[0, :, :].shape)
                af_night[:, :] = np.logical_or(array_night[0, :, :] > th_n[i][0], array_night[1, :, :] > th_n[i][1])
                af_img_night = af_night
                af_img_night[np.logical_not(af_img_night[:, :])] = np.nan
                plt.subplot(223)
                plt.title('viirs af night')
                plt.imshow(array_night[0, :, :])
                plt.imshow(af_img_night, cmap='hsv', interpolation='nearest')
                plt.subplot(224)
                plt.title('Night original')
                plt.imshow(array_night[0, :, :])
                plt.savefig(f'{save_path}_figure/{location}_{i}.png')
                plt.show()
                plt.close()

                # array = self.standardization(array_day)
                col_start = int(array_day.shape[2]//2 - 128)
                row_start = int(array_day.shape[1]//2 - 128)
                array = np.concatenate((array_day[:6,...], array_night, np.logical_or(np.nan_to_num(af[np.newaxis, :, :]), np.nan_to_num(af_night[np.newaxis, :, :]))))
                plt.imshow(np.logical_or(np.nan_to_num(af[np.newaxis, :, :]), np.nan_to_num(af_night[np.newaxis, :, :]))[0,...])
                plt.savefig(f'{save_path}_figure/af_label_{location}_{i}.png')
                plt.show()
                array = array[:, row_start:row_start + image_size[0], col_start:col_start + image_size[1]]
                output_array[i, :, :array.shape[1], :array.shape[2]] = np.nan_to_num(array)
                print(output_array.shape)
            array_stack.append(output_array)
        
        output_array_stacked = np.stack(array_stack, axis=0)
        print(output_array_stacked.shape)
        np.save(save_path + file_name, output_array_stacked[:,:,:-1,:,:].astype(np.float))
        np.save(save_path + file_name.replace('img', 'label'), output_array_stacked[:,:,-1,:,:].astype(np.float))

    def af_seq_tokenizing_and_test_slicing(self, location, modes, ts_length, interval, usecase, root_path, save_path):
        window_size = 1
        print(location)
        if location in ['val', 'train']:
            root_path = f'{root_path}/dataset_{location}'
            save_path = f'{root_path}/dataset_{location}'
            tokenized_array = np.load(os.path.join(root_path, f'af_{location}_img_seqtoseq_alll_{ts_length}i_{interval}.npy')).transpose((0, 3, 4, 2, 1))
            tokenized_label = np.load(os.path.join(root_path, f'af_{location}_label_seqtoseq_alll_{ts_length}i_{interval}.npy')).transpose((0, 3, 4, 2, 1))
            tokenized_label = tokenized_label[..., 2]
        else:
            root_path = '/home/z/h/zhao2/CalFireMonitoring/data_train_proj2'
            save_path = '/home/z/h/zhao2/TS-SatFire/dataset/dataset_test'
            tokenized_array = np.load(os.path.join(root_path, f'af_{location}_img.npy')).transpose((0, 3, 4, 1, 2))
            tokenized_label = np.load(os.path.join(root_path, f'af_{location}_label.npy')).transpose((0, 2, 3, 1))
        if tokenized_array.shape[-2]>=ts_length:
            array_concat = []
            label_concat = []
            for i in range(0, tokenized_array.shape[-2], interval):
            # lb = (tokenized_array.shape[-2]-ts_length)//2
            # rb = (tokenized_array.shape[-2]+ts_length)//2
                if i+ts_length > tokenized_array.shape[-2]:
                    array_concat.append(tokenized_array[:,:,:,tokenized_array.shape[-2]-ts_length:tokenized_array.shape[-2],:])
                    label_concat.append(tokenized_label[:,:,:,tokenized_array.shape[-2]-ts_length:tokenized_array.shape[-2]])
                else:
                    array_concat.append(tokenized_array[:,:,:,i:i+ts_length,:])
                    label_concat.append(tokenized_label[:,:,:,i:i+ts_length])
            tokenized_array = np.concatenate(array_concat, axis=0)
            tokenized_label = np.concatenate(label_concat, axis=0)
            # tokenized_array = tokenized_array[:,:,:,lb:rb,:]
            # tokenized_label = tokenized_label[:,:,:,lb:rb]
        if usecase == 'temp':
            print('tokenizing')
            tokenized_array = np.nan_to_num(tokenized_array).reshape(-1,tokenized_array.shape[-2],tokenized_array.shape[-1])
            tokenized_label = np.nan_to_num(tokenized_label).reshape(-1,tokenized_label.shape[-1])
            print(tokenized_array.shape)
            print(tokenized_label.shape)
            np.save(os.path.join(save_path,f'af_{location}_img_seqtoseq_l{ts_length}_w{window_size}.npy'), tokenized_array)
            np.save(os.path.join(save_path,f'af_{location}_label_seqtoseq_l{ts_length}_w{window_size}.npy'), tokenized_label)
        else:
            img_array = np.nan_to_num(tokenized_array.transpose((0,4,3,1,2)))
            img_label = np.nan_to_num(tokenized_label.transpose((0,3,1,2)))
            img_label = np.repeat(img_label[:,np.newaxis,:,:,:], 3, axis=1)
            # img_array = img_array[:,:,np.newaxis,:,:]
            # img_label = img_label[:,:,np.newaxis,:,:]
            print(img_array.shape)
            print(img_label.shape)
            np.save(os.path.join(save_path,f'af_{location}_img_seqtoseql_{ts_length}i_{interval}.npy'), img_array.astype(np.float32))
            np.save(os.path.join(save_path,f'af_{location}_label_seqtoseql_{ts_length}i_{interval}.npy'), img_label.astype(np.float32))