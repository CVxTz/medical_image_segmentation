import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from aug_utils import random_augmentation
from random import randint


def plot_figures(names, figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(names):
        img = np.squeeze(figures[title])
        if len(img.shape)==2:
            axeslist.ravel()[ind].imshow(img, cmap=plt.gray())#, cmap=plt.gray()
        else:
            axeslist.ravel()[ind].imshow(img)


        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional

    plt.show()


def read_input(path):
    x = Image.open(path)
    return np.array(x)/255.


def read_gt(path):
    y = Image.open(path)
    return np.array(y)/255.


def random_crop(img, mask, crop_size=64):
    imgheight= img.shape[0]
    imgwidth = img.shape[1]
    i = randint(0, imgheight-crop_size)
    j = randint(0, imgwidth-crop_size)

    return img[i:(i+crop_size), j:(j+crop_size), :], mask[i:(i+crop_size), j:(j+crop_size)]

figures = {}
gt_1 = read_gt('../input/DRIVE/training/1st_manual/23_manual1.gif')
img_1 = read_gt('../input/DRIVE/training/images/23_training.tif')
img_1, gt_1 = random_crop(img_1, gt_1, crop_size = 64)
print(img_1.shape, gt_1.shape)
figures['Input 1'] = img_1
figures['Ground Truth 1'] = gt_1

gt_2 = read_gt('../input/DRIVE/training/1st_manual/22_manual1.gif')
img_2 = read_gt('../input/DRIVE/training/images/22_training.tif')
figures['Input 2'] = img_2
figures['Ground Truth 2'] = gt_2

gt_3 = read_gt('../input/DRIVE/training/1st_manual/21_manual1.gif')
img_3 = read_gt('../input/DRIVE/training/images/21_training.tif')
figures['Input 3'] = img_3
figures['Ground Truth 3'] = gt_3
print(gt_3.shape)

# generation of a dictionary of (title, images)
names = ['Input 1', 'Input 2', 'Input 3', 'Ground Truth 1', 'Ground Truth 2', 'Ground Truth 3']

# plot of the images in a figure, with 2 rows and 3 columns
plot_figures(names, figures, 2, 3)