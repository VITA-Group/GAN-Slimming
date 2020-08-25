# GAN-Slimming

## Overview:

## Training:
### 1. Download dataset:
```
./download_dataset <dataset_name>
```
This will download the dataset to folder `datasets/<dataset_name>` (e.g., `datasets/summer2winter_yosemite`).

### 2. Train origianl dense CycleGAN and generate style stransfer results on training set:
Use the [offcial CycleGAN codes](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to train origianl dense CycleGAN and generate style stransfer results on training set.
Put the style transfer results to folder `train_set_result/<dataset_name>`.
For example, `train_set_result/summer2winter_yosemite/B/2009-12-06 06:58:39_fake.png` is the fake winter image transfered from the real summer image `datasets/summer2winter_yosemite/A/2009-12-06 06:58:39.png` using the orignal dense CycleGAN.

### 3. Compress
GS-32:
```
python gs.py --rho 0.01 --dataset <dataset_name> --task <task_name>
```

GS-8:
```
python gs.py --rho 0.01 --quant --dataset <dataset_name> --task <task_name>
```

The training results (checkpoints, loss curves, etc.) will be saved in `results/<dataset_name>/<task_name>`.


### 4. Extract subnetwork obtained by GS:
```
python extract_subnet.py --dataset <dataset_name> --task <task_name> --model_str <model_str> 
```

Finetune subnetwork:
```
python finetune.py --dataset <dataset_name> --task <task_name> --base_model_str <base_model_str>
```

## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{wang2020GAN,
  title={GAN Slimming: All-in-One GAN Compression by A Unified Optimization Framework},
  author={Wang, Haotao and Gui, Shupeng and Yang, Haichuan and Liu, Ji and Wang, Zhangyang},
  booktitle={ECCV},
  year={2020}
}
```