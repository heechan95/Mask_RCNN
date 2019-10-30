# Kaggle IMaterialist

![Instance Segmentation Sample](maskrcnn.png)

## Solution
This repo is based on Matterport MaskRCNN + DenseCRF for post processing

## DenseCRF
![Dense CRF](densecrf.png)

* Using Dense CRF makes the segmentation look better
* In order to use densecrf, you need to install
* densecrf requires the probability of each pexel belonging to each class. so change some files with the files in the "modified" folder
