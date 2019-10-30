import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 15
#import seaborn as sns
import json
from tqdm import tqdm
import pandas as pd
pd.set_option("display.max_rows", 101)
import glob
from collections import Counter
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


#input_dir = "./advancedML/data/"
'''
One Dataset is around more than 200mb
The size of one batch is mainly determined by batch size, image size, maximum object number
batch size = IMAGES_PER_GPU*GPU_COUNT in config.py
image size is related to MAX_IMG_DIM,MIN_IMG_DIM in config.py
maximum object numbers = MAX_GT_OBJECTS in config.py
suppose dtpye is int8 --> at least batch_size*1024*1024*2*100 = batch_size*200mb

This library use keras' fit_generator with argument multiprocessing=true, workers=cpu_count
This means multiple dataset generators are instanciated by each cpu
each cpu=process keeps queue for pipelining, and we can limit the maxmum number of elements in queue by setting max_queue_size in fit_generator function
*it doesn’t mean the training begins only after the queue is filled. Making the yield super-slow shows this.
--> However, if training(consumer) is slower than generating data, queue gets filled, resulting memory blowing up
**Trade-off: if batch size increases, it can help optimize better(gradienet is more accurate), but it can slow dowin the one training step and make queue filled. On the other hand, if batch size is small, whole training time should be longer due to approximate gradient, but one training step is faster, so it does not make queu filled as quickly as when batch size is bigger.
***multiprocessing=false,workers>1 or workers>cpu count-->need for thread-safe generator
****This library workers=cpu count, so you don't need to worry about thread safety
Anyway, avoiding 'out of memory' you need to change some parameters, and The arguments that I think is better to change is : max_queue_size(defualt 100), workers(less than cpu count), IMAGE_PER_GPU(default 1), GPU_COUNT

Example
IMAGE_PER_GPU=8/CPU=16/max_queue_size=100/GPU_COUNT=1
--> maximum size is at least 16*8*1*100*data ==> 2500G!!!!!
120/


KAGGLE KERNEL
CPU=4/RAM=17G --> CPU specifications
CPU=2/RAM=14G --> GPU specifications
IMAGE_PER_GPU=1/max_queu_size=100
--> maximum size is at least 2*1*1*100*data ==> 40G TT

NASH: CPU 32G/ cores: 8 
--> maximum 8*batch_size*max_queue_size*data-->2G*batch_size*max_queue_size
20/() -> 10 >= batch_size*max_queue_size super safe



'''


def classid2label(class_id):
    category, *attribute = class_id.split("_")
    return category, attribute




def json2df(data):
    df = pd.DataFrame()
    for index, el in enumerate(data):
        for key, val in el.items():
            df.loc[index, key] = val
    return df



input_dir = "advanced_ML/data/"
train_df = pd.read_csv(input_dir + "train.csv")
train_df.head()



with open(input_dir + "label_descriptions.json") as f:
    label_description = json.load(f)
    
print("this dataset info")
print(json.dumps(label_description["info"], indent=2))




category_df = json2df(label_description["categories"])
category_df["id"] = category_df["id"].astype(int)
category_df["level"] = category_df["level"].astype(int)
attribute_df = json2df(label_description["attributes"])
attribute_df["id"] = attribute_df["id"].astype(int)
attribute_df["level"] = attribute_df["level"].astype(int)


print("Category Labels")
#category_df


print("Attribute Labels")
#attribute_df


counter_category = Counter()
counter_attribute = Counter()
for class_id in train_df["ClassId"]:
    category, attribute = classid2label(class_id)
    counter_category.update([category])
    counter_attribute.update(attribute)


category_name_dict = {}
for i in label_description["categories"]:
    category_name_dict[str(i["id"])] = i["name"]
attribute_name_dict = {}
for i in label_description["attributes"]:
    attribute_name_dict[str(i["id"])] = i["name"]



ROOT_DIR="fashion/mrcnn"
DATA_DIR="advanced_ML/data"



sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))



from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log



class FashionDataset(utils.Dataset):
    def load_fashion(self,image_ids=None,num_data=None):
        '''
        add_class --> register 46 classes self.add_class('fashion',i,name)
        image_ids --> unique index of images(name?)
        self.add_image('fashion',image_ids,width,height,annotations)
        width,height --> shape[:2] or extract form dataframe
        annotations --> all collections of annotations for each image
        Todo:
        There are some rows that have height and weight as nan value
        validation option is necessary for training
        '''
        for i,row in category_df.iterrows():
            self.add_class('fashion',i,row['name'])
        
        if image_ids is None:
            image_ids = list(set(train_df['ImageId']))
        
        if num_data is not None:
            random.seed(42)
            random.shuffle(image_ids)
            image_ids=image_ids[:num_data]
            
            
        for i in image_ids:
            Width = train_df[train_df['ImageId']==i]['Width'].reset_index(drop=True)[0]
            Height = train_df[train_df['ImageId']==i]['Height'].reset_index(drop=True)[0]
            self.add_image('fashion',
                           image_id=i,
                           path=DATA_DIR+'/train/'+i,
                           width=Width,
                           height=Height,
                           annotations=train_df[train_df['ImageId']==i])
        
        
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        ImagePath = info['path']
        image = np.asarray(Image.open(ImagePath).convert("RGB"))
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        width=info['width']
        height=info['height']
        
        instance_masks = []
        class_ids = []
        
        for i,annotation in annotations.iterrows():
            class_id=annotation['ClassId']
            class_id=class_id.split('_')[0]
            class_ids.append(class_id)
            rle = annotation['EncodedPixels']
            instance_masks.append(self.rle_to_mask(rle,width,height))
            
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)+1
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(FashionDataset, self).load_mask(image_id)
        
        
    def rle_to_mask(self,rle,width,height):
        mask = np.zeros(width*height,dtype=np.int8)
        pixels_list = list(map(int,rle.split(" ")))
        for i in range(0,len(pixels_list),2):
            start_pixel = pixels_list[i]-1
            num_pixel = pixels_list[i+1]-1
            mask[start_pixel:start_pixel+num_pixel] = 1
        
        mask = mask.reshape((height,width),order='F')
        return mask



image_ids_list = list(set(train_df['ImageId']))

random.seed(42)
random.shuffle(image_ids_list)

val_split = 0.1
split = int((1-val_split)*len(image_ids_list))

train_ids = image_ids_list[:split]
val_ids = image_ids_list[split:]
train_ids = train_ids[:100]
val_ids = val_ids[:100]



#fashion_dataset = FashionDataset()
#fashion_dataset.load_fashion(num_data=100)
#fashion_dataset.prepare()


fashion_dataset_train = FashionDataset()
fashion_dataset_train.load_fashion(train_ids)
fashion_dataset_val = FashionDataset()
fashion_dataset_val.load_fashion(val_ids)



fashion_dataset_train.prepare()
fashion_dataset_val.prepare()

print("dataset prepared")

#print("Image Count: {}".format(len(fashion_dataset.image_ids)))
#print("Class Count: {}".format(fashion_dataset.num_classes))
#for i, info in enumerate(fashion_dataset.class_info):
#    print("{:3}. {:50}".format(i, info['name']))



# Load and display random samples
#image_ids = np.random.choice(fashion_dataset.image_ids, 4)
#for image_id in image_ids:
#    image = fashion_dataset.load_image(image_id)
#    mask, class_ids = fashion_dataset.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, fashion_dataset.class_names)



# Load random image and mask.
#image_id = random.choice(fashion_dataset.image_ids)
#image = fashion_dataset.load_image(image_id)
#mask, class_ids = fashion_dataset.load_mask(image_id)
# Compute Bounding box
#bbox = utils.extract_bboxes(mask)

# Display image and additional stats
#print("image_id ", image_id, fashion_dataset.image_reference(image_id))
#log("image", image)
#log("mask", mask)
#log("class_ids", class_ids)
#log("bbox", bbox)
# Display image and instances
#visualize.display_instances(image, bbox, mask, class_ids, fashion_dataset.class_names)


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 1))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

'''
def rle_to_mask(rle,width,height):
    mask = np.zeros(width*height,dtype=np.int8)
    pixels_list = list(map(int,rle.split(" ")))
    for i in range(0,len(pixels_list),2):
        start_pixel = pixels_list[i]-1
        num_pixel = pixels_list[i+1]-1
        mask[start_pixel:start_pixel+num_pixel] = 1
        
    mask = mask.reshape((height,width),order='F')
    return mask



m=rle_to_mask(sample_rle_encoding,3676,5214)
r=rle_encoding(m)

" ".join([str(e) for e in r]) == sample_rle_encoding
'''


class FashionConfig(Config):

    NAME = "fashion"

    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 46
    STEPS_PER_EPOCH=1000
    
config = FashionConfig()
#config.display()



model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=ROOT_DIR)




WEIGHT_PATH = 'last'
'''
if WEIGHT_PATH == "last":
        # Find last trained weights
        model_path = model.find_last()
elif WEIGHT_PATH == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
else:
        pass
        #model_path = args.model

'''


model_path='fashion/mrcnn/pre-trained/mask_rcnn_fashion_0105.h5'
model.load_weights(model_path, by_name=True)



epochs_stage1_1=110
epochs_stage1_2=120
epochs_stage2_1=130
epochs_stage2_2=140
epochs_stage3_1=150
epochs_stage3_2=160


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(fashion_dataset_train, fashion_dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs_stage1_1,
                layers='heads')

model.train(fashion_dataset_train, fashion_dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs_stage1_2,
                layers='heads')

# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
print("Fine tune Resnet stage 4 and up")
model.train(fashion_dataset_train, fashion_dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs_stage2_1,
            layers='4+')

model.train(fashion_dataset_train, fashion_dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs_stage2_2,
            layers='4+')


# Training - Stage 3
# Fine tune all layers
print("Fine tune all layers")
model.train(fashion_dataset_train, fashion_dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=epochs_stage3_1,
            layers='all')

model.train(fashion_dataset_train, fashion_dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=epochs_stage3_2,
            layers='all')


print("Training Finished")

'''

class InferenceConfig(FashionConfig):
    #GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
    
inference_config = InferenceConfig()
inference_config.display()


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=ROOT_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)



image_ids = train_ids[0]
image=fashion_dataset_train.load_image(0)


result = model.detect([image])


r = result[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            fashion_dataset_train.class_names)


def get_test_filepaths(input_dir):
    jpg_fps = glob.glob(input_dir+'/'+'*.jpg')
    return list(set(jpg_fps))

test_input_dir=os.path.join(DATA_DIR, 'test')
test_fps=get_test_filepaths(test_input_dir)
'''
