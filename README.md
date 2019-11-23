# PCA Dimensionality reduction testing on Image

This project involves application of PCA technique on image data and assessing its performance in terms of information retention and compressibility. This excersise deals with practical step-by-step PCA implementation on Image data. The image data has been chosen over tabular data so that the reader can better understand the working of PCA through image visualization. Technically, an image is a matrix of pixels whose brightness represents the reflectance of surface feature within that pixel. The reflectance value ranges from 0 to 255 for an 8-bit integer image. So the pixels with zero reflectance would appear as black, pixels with value 255 appear as pure white and pixels with value in-between appear in a gray tone. Landsat TM satellite Images, captured over the coastal region of India, have been used in this tutorial. The images are resized to a smaller scale to reduce computational load on the CPU. The image set consists of 7 band images captured across the blue, green, red, near-infrared (NIR) and mid-infrared (MIR) range of the electromagnetic spectrum.

files description:
1. Band{1-7}.jpg - LandSat multispectral Images of reflectance captured across blue till mid-infrared spectrum.
2. PCA_VisualDemo-7bands.ipynb - notebook that has python implementation of PCA along with interesting plots and conclusion.

For deatailed explaination please refer to the below article:
https://towardsdatascience.com/principal-component-analysis-in-depth-understanding-through-image-visualization-892922f77d9f
