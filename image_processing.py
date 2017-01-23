# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:29:14 2014

@author: eejvt
"""

import numpy as np
import scipy
import pylab
import pymorph
import mahotas
from scipy import ndimage

import os
os.chdir('/nfs/see-fs-01_users/eejvt')
'''
import numpy as np
import cv2

cap = cv2.VideoCapture('test.avi')

fgbg = cv2.BackgroundSubtractorMOG()
while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

'''



dna = mahotas.imread('test.JPG')
dna = dna.squeeze()

#dnaf = ndimage.gaussian_filter(dna, 8)
T = mahotas.thresholding.otsu(dna)
labeled,nr_objects = ndimage.label(dna > T)
print nr_objects
pylab.imshow(labeled)
pylab.jet()
pylab.show()
