# USAGE
# python train.py
# import the necessary packages

#from pyimagesearch.dataset_tif import SegmentationDataset
#from pyimagesearch.model import UNet
#from pyimagesearch import config
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

# filter the images with all pixels equal zero or any pixel equals nan
tmp_image = imagePaths.copy()
tmp_mask = maskPaths.copy()

nb_nan = 0
nb_zero = 0

for i, e in enumerate(tqdm(tmp_mask,desc='filtering')):
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

print("filter the {} images with any pixels equal nan".format(nb_nan))
print("filter the {} images with all pixels equal zero".format(nb_zero))

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]
# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w+")
f.truncate(0) # Before write, clear all content
f.write("\n".join(testImages))
f.close()

# define transformations
transforms = transforms.Compose([
    transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()
	])
# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE
	#pin_memory=config.PIN_MEMORY,
	#num_workers=os.cpu_count()
	)
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.BATCH_SIZE
	#pin_memory=config.PIN_MEMORY,
	#num_workers=os.cpu_count()
	)



# initialize our UNet model
unet = UNet().to(config.DEVICE)
# initialize loss function and optimizer
#pos_weight = torch.tensor([100]).cuda()
pos_weight = torch.tensor([100]).to(config.DEVICE) # pos_weight!
lossFunc = BCEWithLogitsLoss(pos_weight=pos_weight)
opt = Adam(unet.parameters(), lr=config.INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}



# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
#for e in range(config.NUM_EPOCHS):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	# loop over the training set
	#for (i, (x, y)) in enumerate(trainLoader):
	for x, y in trainLoader:
		# send the input to the device
		#(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
		x = x.to(config.DEVICE)
		y = y.to(config.DEVICE)
		# perform a forward pass and calculate the training loss
		pred = unet(x)
		loss = lossFunc(pred, y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		# loop over the validation set
		for x, y in testLoader:
			# send the input to the device
			#(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			x = x.to(config.DEVICE)
			y = y.to(config.DEVICE)
			# make the predictions and calculate the validation loss
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))





# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH + '_plotnew.png')
# serialize the model to disk
torch.save(unet, config.MODEL_PATH + '_50epochs.pth')




    