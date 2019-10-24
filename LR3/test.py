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

# 1. создать свою хаар-фичу и ее координаты
# 2. создать несколь РОИ и выбрать лучшие хаар-фичи для каждого

def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img = io.imread(os.path.join(folder,filename))
            if img is not None and img.shape[0] == img.shape[1]:
                img = color.rgb2gray(img)
                images.append(img)
        return images


@delayed
def extract_feature_image(img, feature_type, roi_x=0, roi_y=0, roi_width=100, roi_height=100, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
#    print('ii', ii.shape)
#    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
    return haar_like_feature(ii, roi_x, roi_y, roi_width, roi_height,
                             feature_type=feature_type,
                             feature_coord=feature_coord)
    
def getTrainDataset():
    car_images = load_images_from_folder('cars')
    car_images = np.array(car_images)
    return car_images

def add_coord(coords, x, y):
    for idx, tuples in enumerate(coords):
        for i, t in enumerate(tuples):
            tuples[i] = (t[0] + y, t[1] + x)
        coords[idx] = tuples
    return coords
    
def get_feature_coords(images, roi_x, roi_y, roi_width, roi_height):
    feature_types = [ 'type-3-x', 'type-3-y', 'type-4']
    # Build a computation graph using dask. This allows using multiple CPUs for
    # the computation step
    X = delayed(extract_feature_image(img, feature_types, roi_x, roi_y, roi_width, roi_height)
                for img in images)
    # Compute the result using the "processes" dask backend
    t_start = time()
    X = np.array(X.compute(scheduler='processes'))
    time_full_feature_comp = time() - t_start
    t_size = len(X)
    y = np.array([1] * (t_size-2) + [0]*2)           # TODO:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=t_size-2,
                                                        random_state=0,
                                                        stratify=y)
    
    # Extract all possible features to be able to select the most salient.
    feature_coord, feature_type = \
            haar_like_feature_coord(width=roi_width, height=roi_height,
                                    feature_type=feature_types)
    
    ############################################### LEARNING  
    
    # Train a random forest classifier and check performance
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                                 max_features=100, n_jobs=-1, random_state=0)
    t_start = time()
    clf.fit(X_train, y_train)
    time_full_train = time() - t_start
#    auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    #auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test))
    #print('clf.predict_proba(X_test)', clf.predict_proba(X_test))
    
    
    # Sort features in order of importance, plot six most significant
    idx_sorted = np.argsort(clf.feature_importances_)[::-1]
    
    add_coord(feature_coord[idx_sorted[0]], roi_x, roi_y)
    add_coord(feature_coord[idx_sorted[1]], roi_x, roi_y)
    return (
#            [coord1, coord2],
            [feature_coord[idx_sorted[0]], feature_coord[idx_sorted[1]]],
            [feature_type[idx_sorted[0]], feature_type[idx_sorted[1]]]
            )
    
    
def get_best_feature_coords(train_images):
    coords = np.empty(6,dtype=object)
    types = np.empty(6,dtype=object)
    feature_coords1, feature_types1 = get_feature_coords(train_images, 10, 30, 30, 30)
    feature_coords2, feature_types2 = get_feature_coords(train_images, 40, 40, 30, 30)
    feature_coords3, feature_types3 = get_feature_coords(train_images, 70, 50, 30, 30)
    coords[0] = feature_coords1[0]
    coords[1] = feature_coords1[1]
    coords[2] = feature_coords2[0]
    coords[3] = feature_coords2[1]
    coords[4] = feature_coords3[0]
    coords[5] = feature_coords3[1]
    types[0] = feature_types1[0]
    types[1] = feature_types1[1]
    types[2] = feature_types2[0]
    types[3] = feature_types2[1]
    types[4] = feature_types3[0]
    types[5] = feature_types3[1]
    
    fig, axes = plt.subplots(3, 2)
    for idx, ax in enumerate(axes.ravel()):
        image = train_images[3]
        image = draw_haar_like_feature(image, 0, 0,
                                       train_images.shape[2],
                                       train_images.shape[1],
                                       feature_coord=[coords[idx]])
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('The most important features')
    
    return (coords,types)
    

def get_features(images, coords, types):
    features = []
    for idx, coord in enumerate(coords):
        coord_wrap = np.empty(1,dtype=object)
        coord_wrap[0] = coord
        avg_feature = 0
        for i, img in enumerate(images):
#            img = draw_haar_like_feature(img, 0, 0,
#                                           images.shape[2],
#                                           images.shape[1],
#                                           coord_wrap)
#            io.imshow(img)
#            io.show()
            f1 = delayed(extract_feature_image(images[i], np.array([types[idx]]), feature_coord=coord_wrap))
            f1 = f1.compute(scheduler='processes')
            print(f1)
#            avg_feature += abs(f1[0])
            avg_feature += f1[0]
        avg_feature /= len(images)
        features.append(avg_feature)
    print('avg features', features)
    return features


def get_heuristic_coords(images):
    coords = np.empty(4,dtype=object)
    types = np.empty(4,dtype=object)
    coords[0] = [[(45, 65), (75, 75)], [(50, 76), (75, 85)]]
    types[0] = 'type-2-x'
    coords[1] = [[(35, 20), (55, 30)], [(40, 31), (55, 45)], [(45, 46), (55, 55)]]
    types[1] = 'type-3-y'
    coords[2] = [[(25, 5), (45, 15)], [(35, 16), (55, 25)]]
    types[2] = 'type-2-x'
    coords[3] = [[(15, 5), (25, 95)], [(25, 5), (75, 95)], [(75, 5), (85, 95)]]
    types[3] = 'type-3-x'
#    for i, img in enumerate(images):
#        img = draw_haar_like_feature(img, 0, 0,
#                                       images.shape[2],
#                                       images.shape[1],
#                                       coords)
#        io.imshow(img)
#        io.show()
    return (coords, types)
    
    
def get_test_img():
    img_orig = io.imread(os.path.join('img', '004.jpg'))
    gray_img = color.rgb2gray(img_orig)
    return (img_orig, gray_img)


def detect(img, gray_img, avg_features, coords, types, threshold):
    for x in range(int((gray_img.shape[0] - 100)/10)):
        x = x*10
        for y in range(int((gray_img.shape[1] - 100)/10)):
            y = y*10
            im = gray_img[x:x+100, y:y+100]
            f1 = delayed(extract_feature_image(im, types, feature_coord=coords))
            f1 = f1.compute(scheduler='processes')
#            print('f1', f1)
#            f1 = abs(f1)
#            for i, f in enumerate(f1):
#                f1[i] = abs(f)
#            threshold = [0.1,8,3,3,1,0.1]
            if abs(avg_features[0] - f1[0] ) < threshold[0] \
                and abs(avg_features[1] - f1[1] ) < threshold[1] \
                and abs(avg_features[2] - f1[2] ) < threshold[2] \
                and abs(avg_features[3] - f1[3] ) < threshold[3]:
#                and abs(avg_features[4] - f1[4] ) < threshold[4] 
#                and abs(avg_features[5] - f1[5] ) < threshold[5]:
                print('x', x, 'y', y, 'f1', f1)
    #            img_feat = draw_haar_like_feature(img_orig, x+35, y+35,
    #                                   25,
    #                                   25,
    #                                   coord1)
                rr, cc = draw.circle_perimeter(x+50, y+50, 50)
                img[rr, cc] = (0, 255, 0)
    #            io.imshow(img_feat)
    #            io.show()    
    io.imshow(img)
    io.show()
    
if __name__ == '__main__':  
    train_images = getTrainDataset()
    print(train_images.shape)
#    feature_coords, feature_types = get_best_feature_coords(train_images)
#    feature_coords, feature_types = get_heuristic_coords(train_images)
#    avg_features = get_features(train_images, feature_coords, feature_types)
    img, gray_img = test_img = get_test_img()
##    threshold = [0.1,8,3,3,1,0.1]
##    threshold = [15,30,20,250]
#    threshold = [10,30,20,250]
#    detect(img, gray_img, avg_features, feature_coords, feature_types, threshold)
#    get_heuristic_coords(train_images)
    
    