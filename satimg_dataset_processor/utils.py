import numpy as np
import rasterio
from rasterio.merge import merge
import yaml

class SatProcessingUtils:

    def padding(self, coarse_arr, array_to_be_downsampled):
        array_to_be_downsampled = np.pad(array_to_be_downsampled, ((0, 0), (0, coarse_arr.shape[1] * 2 - array_to_be_downsampled.shape[1]), (0, coarse_arr.shape[2] * 2 - array_to_be_downsampled.shape[2])), 'constant', constant_values = (0, 0))
        return array_to_be_downsampled

    def down_sampling(self, input_arr):
        return np.mean(input_arr)

    def standardization(self, array):
        n_channels = array.shape[0]
        for i in range(n_channels):
            nanmean = np.nanmean(array[i, :, :])
            array[i, :, :] = np.nan_to_num(array[i, :, :], nan=nanmean)
            array[i,:,:] = (array[i,:,:]-array[i,:,:].mean())/array[i,:,:].std()
        return np.nan_to_num(array)

    def normalization(self, array):
        n_channels = array.shape[0]
        for i in range(n_channels):
            array[i,:,:] = (array[i,:,:]-np.nanmin(array[i,:,:]))/(np.nanmax(array[i,:,:])-np.nanmin(array[i,:,:]))
        return np.nan_to_num(array)

    def read_tiff(self, file_path):
        with rasterio.open(file_path, 'r') as reader:
            profile = reader.profile
            tif_as_array = reader.read()
        return tif_as_array, profile

    def write_tiff(self, file_path, arr, profile):
        with rasterio.Env():
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(arr.astype(rasterio.float32))

    def mosaic_geotiffs(self, geotiff_files):
        # Read images and metadata
        src_files = [rasterio.open(file) for file in geotiff_files]

        # Merge images using maximum values for overlapping locations
        mosaic, out_transform = merge(src_files, method="max")

        # Copy metadata from the first file
        out_meta = src_files[0].meta.copy()

        # Update metadata with the mosaic dimensions and transform
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform
        })

        # Close source files
        for src in src_files:
            src.close()

        return mosaic, out_meta
    
    def tokenizing(self, data_path, window_size):
        array = np.load(data_path).transpose((0, 3, 4, 2, 1))
        return array

    def flatten_window(self, array, window_size):
        output_array = np.zeros((array.shape[2], (array.shape[3])*pow(window_size,2)))
        for time in range(array.shape[2]):
            output_array[time, :] = array[:, :, time, :].flatten('F')
        return output_array