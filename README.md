
# EMVP: Embracing Visual Foundation Model for Visual Place Recognition with Centroid-Free Probing
The official code of Neurips 2024 paper "EMVP: Embracing Visual Foundation Model for Visual Place Recognition with Centroid-Free Probing".
## Setup

```bash
conda env create -f environment.yml
```

## Dataset

For training, download [GSV-Cities](https://github.com/amaralibey/gsv-cities) dataset. For evaluation download the desired datasets ([MSLS](https://github.com/FrederikWarburg/mapillary_sls), [NordLand](https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W), [SPED](https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W), or [Pittsburgh](https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/))

Note that, the datasets have to be stored in the
"data" folder of current directory. Otherwise, you can replace DATASET_ROOT with your path in the GSVCitiesDataset.py, MapillaryDataset.py, PittsburgDataset.py, NordlandDataset.py, and SPEDDataset.py files, respectively.
## Train

```bash
bash train.sh
```

## Evaluation

```bash
python3 eval_emvp.py --ckpt_path 'weights/emvp.ckpt' --image_size 322 322 --batch_size 256 --val_datasets MSLS Norrdland
```

## Acknowledgements
This code is based on the following works:
 - [MixVPR](https://github.com/amaralibey/MixVPR)
 - [GSV-Cities](https://github.com/amaralibey/gsv-cities)
 - [DINOv2](https://github.com/facebookresearch/dinov2)
 - [SALAD](https://github.com/serizba/salad)