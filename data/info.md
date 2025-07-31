train: original dataset for training
eval: original test dataset

DATA PROCESSING PIPELINE:
1. resized: resized images
2. thresholded: thresholded images through otsu's method
3. cleaned: cleaned images (agressive volume analysis and contour count) (~10 % examples removed)
4. Data augmentation (10 generated images with slight shifts on rotation and position)