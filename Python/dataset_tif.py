# import the necessary packages
from torch.utils.data import Dataset
from torch import permute
from torch import nan_to_num
from torchvision.transforms import ToTensor
import rasterio
import cv2
import torch
import torchvision


class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
		
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
		
	def __getitem__(self, idx):
		# grab the image path from the current index
		#image = rasterio.open(self.imagePaths[idx])
		#image = image.read()
		#image = ToTensor()(image)
		#image = permute(image, (1,2,0))
		
		#mask = rasterio.open(self.maskPaths[idx])
		#mask = mask.read()
		#mask = ToTensor()(mask)
		#nan_to_num(mask, nan=0.0)
		#mask = permute(mask, (1,2,0))
		
		# using cv2 instead of rasterio to load the image
		image = cv2.imread(self.imagePaths[idx], -1)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = torchvision.transforms.ToTensor()(image)
		
		mask = cv2.imread(self.maskPaths[idx], -1)
		mask = torchvision.transforms.ToTensor()(mask)
		
		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)
		# return a tuple of the image and its mask
		return image, mask