import cv2
import numpy as np


imgpath='/mnt/disk2/std2021/hejiabang-data/OCR/attention_img/AttentionData/59041171_106970752.jpg'

img=cv2.imread(imgpath)

cv2.imshow("img",img)

cv2.waitKey(0)
cv2.destroyWindow()