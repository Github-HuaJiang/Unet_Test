# Unet_Test
## Main Changes(in Python file)
### 1.dataset_tif.py(modified): 
Change the way of reading data by using the cv2 package instead of the rasterio package (essentially, there won't be significant differences; the only difference lies in the shape of the data under different reading modes).
### 2.train.py(modified):
a. Add a step for data filtering, excluding images with pixels containing NaN values or images with all zero pixels from the training data.
![code for filtering images]()
