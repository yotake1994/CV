import cv2
import numpy as np


imgrgb1 = cv2.imread('me1.jpeg')
imggray1 = cv2.cvtColor(imgrgb1, cv2.COLOR_BGR2GRAY)
imgrgb2 = cv2.imread('me2.jpeg')
imggray2 = cv2.cvtColor(imgrgb2, cv2.COLOR_BGR2GRAY)

template = cv2.imread('face.jpeg',0)
w, h = template.shape[::-1]
res1 = cv2.matchTemplate(imggray1, template, cv2.TM_CCOEFF_NORMED)
res2 = cv2.matchTemplate(imggray2, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.5
loc1 = np.where(res1 >= threshold)
loc2= np.where(res2 >= threshold)

for pt in zip(*loc1[::-1]):
    cv2.rectangle(imgrgb1, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
for pt in zip(*loc2[::-1]):
    cv2.rectangle(imgrgb2, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Detected',imgrgb1)
cv2.imshow('dectected2', imgrgb2)
cv2.waitKey(0)
cv2.destroyAllWindows()