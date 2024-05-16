# USAGE
# python predict.py

# import the necessary packages
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import cv2
import os
from imutils import paths
from model import UNet
# import rasterio   
# from rasterio.transform import from_origin
# from torch import permute
# from torchvision.transforms import ToTensor
# from torchvision import transforms

def scale_per(matrix):
    w,h,d = matrix.shape
    matrix = np.reshape(matrix, [w*h,d]).astype(np.float64)
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w,h,d])
    matrix = matrix.clip(0,1)
    return matrix
    
def prepare_plot(origImage, origMask, predMask, img_name):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	plt.savefig(os.path.join(os.path.join(config.PRED_OUTPUT, 'preds'), img_name))
	#figure.show()
    
def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	
	# turn off gradient tracking
	with torch.no_grad():
	    image = cv2.imread(imagePath, -1)
	    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    np_image = image.copy()
	    
	    image = torchvision.transforms.ToTensor()(image)
	    image = torchvision.transforms.ToPILImage()(image)
	    image = torchvision.transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))(image)
	    image = torchvision.transforms.ToTensor()(image)
	    image = image.unsqueeze(0)
	    image = image.to(config.DEVICE)
	    
	    img_name = imagePath.split(os.path.sep)[-1]
	    gt_path = os.path.join(os.path.join(config.PRED_OUTPUT, 'masks'), img_name)
	    gt_path = gt_path.replace('img', 'mask')
	    gt = cv2.imread(gt_path, -1)
	    np_gt = np.array(gt)[...,np.newaxis]
	    
	    pred = model(image).squeeze() # Hua: 4->2
	    pred = torch.sigmoid(pred)
	    pred = pred.cpu().numpy()
	    #pred = (pred > config.THRESHOLD)
	    #print(pred.shape)
	    np_pred = pred[...,np.newaxis]
	    
	    prepare_plot(scale_per(np_image), scale_per(np_gt), scale_per(np_pred), 'nofilter'+img_name)
	    
	    #--------------------------------------------
# 	    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         np_image = image.copy()
        
#         image = torchvision.transforms.ToTensor()(image)
#         image = torchvision.transforms.ToPILImage()(image)
#         image = torchvision.transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))(image)
#         image = torchvision.ToTensor()(image)
#         image = image.unsqueeze(0)
#         image = image.to(config.DEVICE)
        
#         img_name = imagePath.split(os.path.sep)[-1]
#         gt_path = os.path.join(os.path.join(config.PRED_OUTPUT, 'masks'), img_name)
#         gt_path = gt_path.replace('img', 'mask')
#         gt = cv2.imread(gt_path, '-1')
#         np_gt = np.array(gt)[...,np.newaxis]
        
# 		pred_path = os.path.join(os.path.join(config.PRED_OUTPUT, 'preds'), img_name)
# 		pred_path = pred_path.replace('img','pred')
# 		print(gt_path)
# 		print(pred_path)
        
#         pred = model(image).squeeze() # Hua: 4->2
#         pred = torch.sigmoid(pred)
#         pred = pred.cpu().numpy()
#         pred = (pred > config.THRESHOLD)
#         print(pred.shape)
#         np_pred = pred[...,np.newaxis]
        
#         prepare_plot(scale_per(np_img), scale_per(np_gt), scale_per(np_pred), img_name)

print("[INFO] loading up test image paths...")
imagePaths = sorted(list(paths.list_images(os.path.join(config.PRED_OUTPUT, 'imgs'))))

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH + '_50epochs.pth').to(config.DEVICE)

# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, path) 
