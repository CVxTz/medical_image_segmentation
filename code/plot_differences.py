import cv2
import numpy as np
from PIL import Image

image_idx = "01"

gt_path = "/home/jenazzad/PycharmProjects/medical_image_segmentation/input/DRIVE/test/1st_manual/%s_manual1.gif"%image_idx
predicted_path = "/home/jenazzad/PycharmProjects/medical_image_segmentation/output/baseline_unet_aug_do_0.1_activation" \
                 "_ReLU_test/%s_test.tif"%image_idx


gt = np.array(Image.open(gt_path))
predicted = np.array(Image.open(predicted_path))
predicted = (predicted>255/2).astype(int)*255


diff = np.zeros(gt.shape+(3,))
print(diff.shape, gt.shape)

diff[:, :, 0] = gt
diff[:, :, 1] = gt
diff[:, :, 2] = predicted

cv2.imwrite("../output/%s_diff.jpg"%image_idx , diff)