#https://scikit-image.org/docs/dev/auto_examples/applications/plot_haar_extraction_selection_classification.html#sphx-glr-auto-examples-applications-plot-haar-extraction-selection-classification-py

from __future__ import division, print_function
from time import time

import numpy as np
import matplotlib.pyplot as plt

from dask import delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

import os
from skimage import io, filters, color, draw

roi_width = 20
roi_height = 20
roi_x = 20
roi_y = 20

def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img = io.imread(os.path.join(folder,filename))
            if img is not None and img.shape[0] == img.shape[1]:
                print('load_images_from_folder', img.shape)
                img = color.rgb2gray(img)
                images.append(img)
        return images


@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
#    print('ii', ii.shape)
#    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
    return haar_like_feature(ii, roi_x, roi_y, roi_width, roi_height,
                             feature_type=feature_type,
                             feature_coord=feature_coord)


############################################### DATA    
# Create car's dataset
car_images = load_images_from_folder('cars')
car_images = np.array(car_images)
#images = lfw_subset()
#print(car_images.shape)
#
# For speed, only extract the two first types of features
feature_types = ['type-3-x', 'type-3-y', 'type-4']



## Build a computation graph using dask. This allows using multiple CPUs for
## the computation step
#X = delayed(extract_feature_image(img, feature_types)
#            for img in car_images)
## Compute the result using the "processes" dask backend
#t_start = time()
#X = np.array(X.compute(scheduler='processes'))
#time_full_feature_comp = time() - t_start
##y = np.array([1] * 100 + [0] * 100)
#y = np.array([1] * 3 + [0]*2)
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=3,
##X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150,
#                                                    random_state=0,
#                                                    stratify=y)
#
## Extract all possible features to be able to select the most salient.
#feature_coord, feature_type = \
#        haar_like_feature_coord(width=25, height=25,
#                                #        haar_like_feature_coord(width=images.shape[2], height=images.shape[1],
#                                feature_type=feature_types)
#
#print('X', X)
#print('X[0] shape', X.shape)
#print('X_train', X_train)
#print('X_test', X_test)
#print('y_train', y_train)
#print('y_test', y_test)
#print('feature_coord', feature_coord)
#print('feature_type', feature_type)
#
#
################################################ LEARNING  
#
## Train a random forest classifier and check performance
#clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
#                             max_features=100, n_jobs=-1, random_state=0)
#t_start = time()
#clf.fit(X_train, y_train)
#time_full_train = time() - t_start
##auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
##auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test))
##print('clf.predict_proba(X_test)', clf.predict_proba(X_test))
#
#
## Sort features in order of importance, plot six most significant
#idx_sorted = np.argsort(clf.feature_importances_)[::-1]
#print('clf.feature_importances_', clf.feature_importances_)
#print('idx_sorted[0] feature', clf.feature_importances_[idx_sorted[0]])
#print('idx_sorted[1] feature', clf.feature_importances_[idx_sorted[1]])
#print('idx_sorted', idx_sorted)
##print(clf.feature_importances_)
#
#
#fig, axes = plt.subplots(3, 2)
#for idx, ax in enumerate(axes.ravel()):
#    image = car_images[0]
#    print(idx, [feature_coord[idx_sorted[idx]]])
#    image = draw_haar_like_feature(image, roi_x, roi_y,
##                                   images.shape[2],
#                                   roi_width,
##                                   images.shape[1],
#                                   roi_height,
#                                   [feature_coord[idx_sorted[idx]]])
#    ax.imshow(image)
#    ax.set_xticks([])
#    ax.set_yticks([])
#fig.suptitle('The most important features')

#print('feature_coord[idx_sorted[0]]', feature_coord[idx_sorted[0]])
#coord1=np.empty(3,dtype=object)
#feature_type1 = np.empty(3,dtype=object)
#for i,v in enumerate(coord1): coord1[i]=feature_coord[idx_sorted[i]]
#for i,v in enumerate(feature_type1): feature_type1[i]=feature_type[idx_sorted[i]]
coord1 = np.empty(1,dtype=object)
feature_type1 = np.empty(1,dtype=object)
coord1[0] = [[(20, 5), (80, 30)], [(20, 31), (80, 70)], [(20, 71), (80, 90)]]
#coord1[1] = [[(20, 4), (80, 20)], [(20, 21), (80, 40)], [(20, 41), (80, 60)]]

feature_type1[0] = 'type-3-y'
#feature_type1[1] = 'type-3-y'
roi_width = 100
roi_height = 100
roi_x = 0
roi_y = 0

avg_feature = 0
for i, img in enumerate(car_images):
    img = draw_haar_like_feature(img, roi_x, roi_y,
                                   roi_width,
                                   roi_height,
                                   coord1)
    io.imshow(img)
    io.show()
    f1 = delayed(extract_feature_image(car_images[i], feature_type1, coord1))
    f1 = f1.compute(scheduler='processes')
    print(f1)
    avg_feature += abs(f1[0])
avg_feature /= len(car_images)
print('avg_feature', avg_feature)


############################################### TESTING
img_orig = io.imread(os.path.join('img', '004.jpg'))
img = color.rgb2gray(img_orig)
#img_feat = draw_haar_like_feature(img, 35, 35,
#                                   25,
#                                   25,
#                                   coord1)
#io.imshow(img_feat)
#io.show()
#f1 = delayed(extract_feature_image(img, np.array(['type-2-x']), coord1))
#print(f1.compute(scheduler='processes'))

#img_feat = img_orig
#cars = []
#for x in range(int((img.shape[0] - 100)/10)):
#    x = x*10
#    for y in range(int((img.shape[1] - 100)/10)):
#        y = y*10
#        im = img[x:x+100, y:y+100]
##        io.imshow(im)
##        io.show()
#        f1 = delayed(extract_feature_image(im, feature_type1, coord1))
#        f1 = f1.compute(scheduler='processes')[0]
#        f1 = abs(f1)
#        threshold = 50
##        print('f1 ',f1)
##        print('avg_feature - f1 ', avg_feature - f1)
#        if abs(avg_feature - f1) < threshold:
#            print('x', x, 'y', y, 'f1', f1)
##            img_feat = draw_haar_like_feature(img_orig, x+35, y+35,
##                                   25,
##                                   25,
##                                   coord1)
##            rr, cc = draw.rectangle_perimeter(start=(x+35, y+35), end=(x+100, y+100), shape=im.shape)
#            rr, cc = draw.circle_perimeter(x+50, y+50, 50)
#            img_feat[rr, cc] = (0, 255, 0)
##            io.imshow(img_feat)
##            io.show()
#
#io.imshow(img_feat)
#io.show()

    