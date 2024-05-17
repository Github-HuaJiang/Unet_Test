# Unet_Test
## Main Changes(in Python file)
### 1.dataset_tif.py(modified): 
Change the way of reading data by using the cv2 package instead of the rasterio package (essentially, there won't be significant differences; the only difference lies in the shape of the data under different reading modes).
### 2.train.py(modified):
a. Add a step for data filtering, excluding images with pixels containing NaN values or images with all zero pixels from the training data. Because it may cause calculation problems during forward or backward process.

![code for filtering images](https://github.com/Github-HuaJiang/Unet_Test/blob/main/snapshot/datafilter.png)

b. set the hyperparemeter **pos_weight=100** (used for solving the class imbalance), since the pixels representing trees (label 1) are far fewer than background pixels (label 0), the pos_weight parameter can be set higher to emphasize the loss of positive labels.
### 3.predict_test.py(modified):

Before visualization, performing some data processing, such as removing outliers at both ends of each channel of the image or mask

![code for removing outliers](https://github.com/Github-HuaJiang/Unet_Test/blob/main/snapshot/visualization.png)

## Usage(Colab)
### Download
First download the project on the Colab. Then import the project and install corresponding libraries through following command lines.
![command lines]()
