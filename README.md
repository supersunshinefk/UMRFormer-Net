# UMRFormer-Net: 3D Pancreas and Tumor Segmentation Method Based on A Volumetric Transformer
We proposed a novel transformer block that used a combination of MHSA and RDCB as the basic unit to deep integrate both long-range and local spatial information. The MRFormer block was embedded between the encoder and decoder in UNet at the last two layers which was named as UMRFormer-Net. And we applied it in pancreas and pancreas tumor segmentation to help capture more effective feature context information. 

Parts of codes are borrowed from nn-UNet.

![UMRFormer-Net Architecture](/UMRFormer-Net.png)

## Installation
#### 1、System requirements
This software was originally designed and run on a system running Ubuntu 18.04, with Python 3.8, PyTorch 1.8.1, and CUDA 10.1. For a full list of software packages and version numbers, see the Conda environment file `environment.yml`. 

This software leverages graphical processing units (GPUs) to accelerate neural network training and evaluation; systems lacking a suitable GPU will likely take an extremely long time to train or evaluate models. The software was tested with the NVIDIA RTX 3090 GPU, though we anticipate that other GPUs will also work, provided that the unit offers sufficient memory. 

## Training
#### 1、Datasets
Datasets can be downloaded at the following links:

MSD PANCREAS : http://medicaldecathlon.com/

#### 2、Setting up the datasets
While we provide code to load data for training a deep-learning model, you will first need to download images from the above repositories. Regarding the format setting and related preprocessing of the dataset, we operate based on UMRFormer_Net, so I won’t go into details here. You can see [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for specific operations. 

Regarding the downloaded data, I will not introduce too much here, you can go to the corresponding website to view it. Organize the downloaded DataProcessed as follows:

After that, you can preprocess the data using:
```
UMRFormer_convert_decathlon_task -i ../DATASET/UMRFormer_raw/nnFormer_raw_data/Task07_MSD
UMRFormer_plan_and_preprocess -t 7
```

#### 3 Training and Testing the models
##### A. Use the best model we have trained to infer the test set

- MSD

Inference
```
UMRFormer_predict -i ../DATASET/UMRFormer_raw/UMRFormer_raw_data/Task007_MSD/imagesTs -o ../DATASET/UMRFormer_raw/UMRFormer_raw_data/Task007_MSD/inferTs/output -m 3d_fullres -f 0 -t 7 -chk model_best -tr UMRFormerTrainerV2
```

Calculate DICE

```
python ./UMRFormer_Net/MSD_dice/inference.py
```

##### B. The complete process of retraining the model and inference
##### (1).Training 
- MSD
```
UMRFormer_train 3d_fullres UMRFormerTrainerV2_MSD 7 0 
```
##### (2).Evaluating the models
- MSD

Inference
```
UMRFormer_predict -i ../DATASET/UMRFormer_raw/UMRFormer_raw_data/Task007_MSD/imagesTs -o ../DATASET/UMRFormer_raw/UMRFormer_raw_data/Task007_MSD/inferTs/output -m 3d_fullres -f 0 -t 7 -chk model_best -tr UMRFormerTrainerV2
```

Calculate DICE

```
python ./UMRFormer_Net/MSD_dice/inference.py
```

