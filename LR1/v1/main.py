#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:21:43 2019

@author: nozim

http://scikit-image.org/docs/stable/api/api.html
http://scikit-image.org/docs/stable/api/skimage.transform.html
http://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.EllipseModel

"""

from lr1 import LR1, model
import numpy as np
#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=10)
import matplotlib.pyplot as plt
from skimage.feature import ORB, match_descriptors, local_binary_pattern, hog, peak_local_max, shape_index
from skimage.color import rgb2gray
from skimage.feature import corner_foerstner, corner_peaks
from skimage.measure import perimeter, moments, shannon_entropy
from skimage import morphology

LR1 = LR1()

trainApple = LR1.loadTrainApple()
trainPapaya = LR1.loadTrainPapaya()


#image = LR1.getOneImg()
#print(image.max())
#image = LR1.scharr(image)
#LR1.imshow(image)
##LR1.arrayInfo(image)
#print('Width: ', LR1.getFruitWidth(image))
#print('Height: ', LR1.getFruitHeight(image))
#print('Perimeter: ', LR1.getFruitPerimeter(image))
#image = (image > 0.05).astype(int)
#print(image.min())
#LR1.imshow(image)
#print('Width: ', LR1.getFruitWidth(image))
#print('Height: ', LR1.getFruitHeight(image))
#print('Perimeter: ', LR1.getFruitPerimeter(image))
#print(LR1.perim(image))

#LR1.imshow(image)

model = model(LR1)
model.train(trainApple, 'apple')
model.train(trainPapaya, 'papaya')


#image = LR1.getOneTestPapaya()
image = LR1.getOneTestApple()

LR1.imshow(image)
image = LR1.scharr(image)
#image = (image > 0.05).astype(int)
LR1.imshow(image)

LR1.imshow(local_binary_pattern(image,10 ,20)) # no
hog = hog(image)
#print(hog(image))
img1 = shape_index(image, sigma=3, cval=0)
img1 = (img1 > 0.4).astype(int)
LR1.imshow(img1)
print(perimeter(img1))
print(moments(img1))
print(shannon_entropy(img1))
#LR1.imshow(morphology.remove_small_objects(img1, min_size=100, connectivity=10))
#LR1.imshow(morphology.remove_small_holes(img1))
#LR1.imshow(morphology.convex_hull_image(img1))
#LR1.imshow(morphology.convex_hull_object(img1))
#LR1.imshow(morphology.remove_small_objects(img1))
#LR1.imshow(morphology.thin(img1))
LR1.imshow(morphology.skeletonize(img1))
#LR1.imshow(morphology.closing(img1))
#LR1.imshow(morphology.dilation(img1))
#LR1.imshow(morphology.erosion(img1))
print(morphology.black_tophat(img1))
#LR1.imshow(hog(image))


#print("feature ", model.getFeature(image))
#prediction = model.predictOne(image)
#print("prediction: ", prediction)

#classes = model.train([10,20,30], "apple")
#classes = model.train([5,5,5], "apple")
#classes = model.train([5,5,5], "papaya")
print('\n')
#print("meanClasses", model.getTrainedClasses())
#print("classes", model.classes)
print('\n')


model.predict(LR1.loadTestApple(), "apple")
model.predict(LR1.loadTestPapaya(), "papaya")


    
    
    
