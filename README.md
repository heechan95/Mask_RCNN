# Kaggle IMaterialist

![Instance Segmentation Sample](maskrcnn.png)

## Solution
This repo is based on Matterport MaskRCNN + DenseCRF for post processing

## DenseCRF
![Dense CRF](densecrf.png)

* Using Dense CRF makes the segmentation look better
* In order to use densecrf, you need to install the repo in the link https://github.com/lucasb-eyer/pydensecrf
* Densecrf requires the probability of each pexel belonging to each class, whereas this library only returns 1 or 0
* so change some files with the files in the "modified" folder
