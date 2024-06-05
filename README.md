# TS-SatFire Time-Series Multi-Task Satellite Imagery Dataset for Wildfire Detection and Prediction

## Repo Structure
    .
    ├── figures                                   # Analysis file for VIIRS Band I4 Data
    ├── data_processor                            # Data processor from VIIRS Geotiff to time-series and tokenized time-series 
    │   ├── satimg_dataset_processor.py           # Entry to generate time-series from VIIRS Geotiff
    │   ├── data_generator_tf.py                  # dataloader for temporal models implemented in tensorflow
    │   ├── data_generator_torch.py               # dataloader for spatial models and spatial-temporal models
    ├── spatial_models                            # All the spatial models and spatial-temporal models (UNETR SwinUNETR) needed for the project 
    ├── temporal_models                           # All the temporal models (RNN LSTM and T4Fire) needed for the project
    ├── dataset_gen_afba.py                       # Main Entry to generate train/val/test dataset used for active fire detection (AF) and burned area mapping (BA)
    ├── dataset_gen_pred.py                       # Main Entry to generate train/val/test dataset used for wildfire prediction
    ├── run_seq_model.py                          # Main Entry to run spatial models
    ├── run_spatial_model.py                      # Main Entry to run temporal models
    ├── run_spatial_temp_model.py                 # Main Entry to run spatial-temporal models for AF and BA
    ├── run_spatial_temp_model_pred.py            # Main Entry to run spatial-temporal models for prediction task
    └── README.md

## Abstract

We introduce a comprehensive multi-temporal remote sensing dataset covering the entire life cycle of wildfires for active fire detection, daily wildfire monitoring and next-day wildfire prediction. This multi-task dataset comprises 179 wildfire events recorded majority in the US between January 2017 and October 2021. For each wildfire, images from the beginning until the end of the wildfire are provided. It includes a total of 3552 surface reflectance images along with auxiliary data such as weather, topography, and fuel information. Labels for current active fires (AF) and burned areas (BA) are provided for each image. Manual quality assurance is performed for all AF labels and BA test labels.The dataset sets three distinct tasks: a) active fire detection, b) daily burned area mapping and c) daily wildfire progression prediction. Detection tasks, such as active fire detection and burned area mapping, require pixel-wise classification utilizing multi-spectral, multi-temporal images. Prediction tasks involve learning the underlying physical processes by integrating satellite observations with auxiliary data. The primary objective of this dataset is to stimulate further research in wildfire monitoring, particularly leveraging advanced deep learning models capable of effectively processing multi-temporal, multi-spectral images to detect fires and accurately predict fire progression. The benchmarks for detection and prediction tasks indicate that utilizing both spatial and temporal information is crucial for this dataset.

## Spectral bands used in the dataset
![Alt text](figures/flowchart.svg?raw=true "Dataset Setup")
![Alt text](figures/Bands.png?raw=true "Dataset Channels")


## Preparing the dataset
The dataset in GeoTIFF format can be downloaded from the Google Gdrive link below:
[https://drive.google.com/drive/folders/1ea49dNLa7gqtOTIJ5FCiDExWTFRa7gNQ?usp=sharing]

Prepare the environment
```
conda create --name <env> --file requirements.txt
```

Prepare the Active fire detection dataset and Burned area mapping dataset:
```
python dataset_gen_afba.py -mode (train/val/test) -ts (length of the time-series) -it (interval between each sampling) -uc (ba/af Active fire detection or burned area mapping)
```
Prepare the Fire prediction prediction dataset:
```
python dataset_gen_pred.py -mode (train/val/test) -ts (length of the time-series) -it (interval between each sampling)
```

## Rerun the experiement
```
python run_spatial_temp_model.py -m <model name> -mode <af/ba> -b <batch size> -r <number of run> -lr <learning rate> -nh <hyperparameters of SwinUNETR> -ed <hyperparameters of SwinUNETR and UNETR> -nc <number of input channels> -ts <length of time-series> -it <interval> -test
```

## Author

#### Yu Zhao (zhao2@kth.se), Sebastian Gerard(sgerard@kth.se), Yifang Ban (yifang@kth.se), KTH Royal Institute of Technology, Stockholm, Sweden

## Acknowledgement
#### The research is part of the project ‘Sentinel4Wildfire’ funded by Formas, the Swedish research council for sustainable development and the project ‘EO-AI4Global Change’ funded by Digital Futures.
