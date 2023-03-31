# About this repo
This repo contains code for UCSD CSE 273 Winter 2022 final project
by Zhao Yang <zht004@ucsd.edu>
and Zhengyuan Yang <zhy015@ucsd.edu>
It maining consists of a depth-from-strteo algorithm based on energy minimization.

The stereo matching algorithm is described in [1]. Implementation refers to the source code [2] linked with the original paper.

[1] "Kolmogorov and Zabih’s Graph Cuts Stereo Matching Algorithm" by Vladimir Kolmogorov, Pascal Monasse, Pauline Tan (2014) IPOL, http://www.ipol.im/pub/pre/97/

[2] https://github.com/pmonasse/disparity-with-graph-cuts.git

# Usage
## Kolmogorov and Zabih’s Graph Cuts Stereo Matching Algorithm
The program assumes rectified image pairs with left image named `im0.png` and right image named `im1.png`. The default program to execute `.py` files is assumed to be Python.

To speed up the computation of the KZ's stereo matching algorithm, one can vertically divide the images into strips and run the algorithm on each pair of strips in parallel. To run KZ's stereo matching algorithm on a pair of images, execute the following command in a Windows command line terminal:<br />
 `run_disparity.bat imagefolder batchsize overlapsize totalHeight outputfolder numdisparity`<br />
 - imagefolder - the path to the folder where the rectified image pair is stored
 - batchsize - the height of each strip, set to totalHeight / (#CPU cores) to maximize efficiency
 - overlapsize - the overlap height of two consecutive strips, should be less than batchsize
 - totalHeight - the height of the original image
 - outputfolder - the path to the folder where the disparity maps are stored after computation

 Use `stiching.ipynb` to process the resulting disparity strips. Call `load_partial()` to load disparity strips. Use the same `outputfolder`, `totalHeight`, `batchsize`, and `overlapsize` parameters as in the last step. Then, call `stitch()` to stitch disparity stips into a disparity map which aligns with the right image.

 ## Depth of field blur
Use `dof.ipynb` to apply depth of field blur effect onto a color image with its disparity map. Call `dof_blur()` with the following parameters to generate a blurred image.
- color_image - the RGB image to apply DoF blur effect on
- disparity_map - the disparity map of the RGB iamge, should be aligned with the RGB image
- disp_in_focus - the disparity of the focus plane
- levels - the number of disparity levels; increasing `levels` without changing `blurriness` also increases the blurriness
- blurriness - how strong the blurriness is if not in focus
