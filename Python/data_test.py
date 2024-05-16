from dataset_tif import SegmentationDataset
from model import UNet
import config

import os
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import cv2
import torchvision


# load the image and mask filepaths in a sorted manner
imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

tmp_image = imagePaths.copy()
tmp_mask = maskPaths.copy()

nb_nan = 0
nb_zero = 0


for i, e in enumerate(tmp_mask):
    mask = cv2.imread(e, -1)
    mask = torchvision.transforms.ToTensor()(mask)
    tmp_zeros = torch.zeros_like(mask)

    if torch.isnan(mask).any():
        assert(e.split('/', 3)[-1].split('_', 1)[-1] == tmp_image[i].split('/', 3)[-1].split('_', 1)[-1])
        maskPaths.remove(e)
        imagePaths.remove(tmp_image[i])
        nb_nan += 1
        continue

    if torch.eq(mask, tmp_zeros).all():
        assert(e.split('/', 3)[-1].split('_', 1)[-1] == tmp_image[i].split('/', 3)[-1].split('_', 1)[-1])
        maskPaths.remove(e)
        imagePaths.remove(tmp_image[i])
        nb_zero += 1

print(nb_nan)
print(nb_zero)
        
# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]
# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
# print("[INFO] saving testing image paths...")
# f = open(config.TEST_PATHS, "w")
# f.write("\n".join(testImages))
# f.close()

# define transformations
transforms = transforms.Compose([
    transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])
# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainMasks, transforms=transforms)
#trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
#testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
#print(f"[INFO] found {len(testDS)} examples in the test set...")

#print(trainDS[0][0].shape, trainDS[0][1].shape)
#create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, 
	#pin_memory=config.PIN_MEMORY,
	#num_workers=os.cpu_count()
	)
# testLoader = DataLoader(testDS, shuffle=False,
# 	batch_size=config.BATCH_SIZE,
# 	#pin_memory=config.PIN_MEMORY,
# 	#num_workers=os.cpu_count()
# 	)
	
for x in trainLoader:
    print(x.shape)

    