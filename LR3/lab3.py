from __future__ import division, print_function

import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.data import lfw_subset
import numpy as np

from time import time

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

def resize19(images):
    for idx, img in enumerate(images):
        image_resized = resize(img, (30, 30), anti_aliasing=True)
        images[idx]=image_resized
    return images

def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img = io.imread(os.path.join(folder,filename))
            if img is not None and img.shape[0] == img.shape[1]:
                img = color.rgb2gray(img)
                images.append(img)
        return images
    
def get_dataset():
    cars_img = load_images_from_folder('cars')
    not_car_img = load_images_from_folder('not_cars')
    images = cars_img + not_car_img
    images = resize19(images)
    images = np.array(images)
    return images

@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)
    
images = get_dataset()
train_img_count = 13
train_size=20
#images = lfw_subset()
#train_img_count = 100
#train_size=150
    
# For speed, only extract the two first types of features
feature_types = ['type-2-x', 'type-2-y', 'type-3-y', 'type-3-y', 'type-4']

# Build a computation graph using dask. This allows using multiple CPUs for
# the computation step
X = delayed(extract_feature_image(img, feature_types)
            for img in images)
# Compute the result using the "processes" dask backend
t_start = time()
X = np.array(X.compute(scheduler='processes'))
time_full_feature_comp = time() - t_start
y = np.array([1] * train_img_count + [0] * train_img_count)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                    random_state=0,
                                                    stratify=y)

# Extract all possible features to be able to select the most salient.
feature_coord, feature_type = \
        haar_like_feature_coord(width=images.shape[2], height=images.shape[1],
                                feature_type=feature_types)

# Train a random forest classifier and check performance
clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=100, n_jobs=-1, random_state=0)
t_start = time()
clf.fit(X_train, y_train)
time_full_train = time() - t_start
auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

# Sort features in order of importance, plot six most significant
idx_sorted = np.argsort(clf.feature_importances_)[::-1]

fig, axes = plt.subplots(3, 2)
for idx, ax in enumerate(axes.ravel()):
    image = images[0]
    image = draw_haar_like_feature(image, 0, 0,
                                   images.shape[2],
                                   images.shape[1],
                                   [feature_coord[idx_sorted[idx]]])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('The most important features')


cdf_feature_importances = np.cumsum(clf.feature_importances_[idx_sorted])
cdf_feature_importances /= np.max(cdf_feature_importances)
sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.7)
sig_feature_percent = round(sig_feature_count /
                            len(cdf_feature_importances) * 100, 1)
print(('{} features, or {}%, account for 70% of branch points in the random '
       'forest.').format(sig_feature_count, sig_feature_percent))

# Select the most informative features
selected_feature_coord = feature_coord[idx_sorted[:sig_feature_count]]
selected_feature_type = feature_type[idx_sorted[:sig_feature_count]]
# Note: we could select those features from the
# original matrix X but we would like to emphasize the usage of `feature_coord`
# and `feature_type` to recompute a subset of desired features.

# Delay the computation and build the graph using dask
X = delayed(extract_feature_image(img, selected_feature_type,
                                  selected_feature_coord)
            for img in images)
# Compute the result using the *threads* backend:
# When computing all features, the Python GIL is acquired to process each ROI,
# and this is where most of the time is spent, so multiprocessing is faster.
# For this small subset, most of the time is spent on the feature computation
# rather than the ROI scanning, and using threaded is *much* faster, because
# we avoid the overhead of launching a new process.
t_start = time()
X = np.array(X.compute(scheduler='threads'))
time_subs_feature_comp = time() - t_start
y = np.array([1] * train_img_count + [0] * train_img_count)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                    random_state=0,
                                                    stratify=y)


t_start = time()
clf.fit(X_train, y_train)
time_subs_train = time() - t_start

auc_subs_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

summary = (('Computing the full feature set took {:.3f}s, plus {:.3f}s '
            'training, for an AUC of {:.2f}. Computing the restricted feature '
            'set took {:.3f}s, plus {:.3f}s training, for an AUC of {:.2f}.')
           .format(time_full_feature_comp, time_full_train, auc_full_features,
                   time_subs_feature_comp, time_subs_train, auc_subs_features))

print(summary)
plt.show()


print('================')

#car = images[0]
#io.imshow(car)
#io.show()
#car_f = extract_feature_image(car, selected_feature_type,
#                                selected_feature_coord)
#car_f = np.array([car_f.compute(scheduler='threads')])
##print(car_f, car_f.shape)
##print(X_test, X_test.shape)
#print('predict', clf.predict(car_f))
#
#not_car = images[20]
#io.imshow(not_car)
#io.show()
#not_car_f = extract_feature_image(not_car, selected_feature_type,
#                                selected_feature_coord)
#not_car_f = np.array([not_car_f.compute(scheduler='threads')])
#print('predict', clf.predict(not_car_f))
#print('predict_proba', clf.predict_proba(X_test))

def get_test_img():
    img_orig = io.imread(os.path.join('img', '004.jpg'))
    gray_img = color.rgb2gray(img_orig)
    return (img_orig, gray_img)

def detect(img, gray_img, clf, coords, types):
    w_size = 30
    for x in range(int((gray_img.shape[0] - w_size)/10)):
        x = x*10
        for y in range(int((gray_img.shape[1] - w_size)/10)):
            y = y*10
            im = gray_img[x:x+w_size, y:y+w_size]
#            im = resize(im, (30, 30), anti_aliasing=True)
            f = extract_feature_image(im, types,
                                coords)
            f = np.array([f.compute(scheduler='threads')])
            p = clf.predict(f)
            if p[0] == 1:
                print('x', x, 'y', y)
                half_size = int(w_size/2)
                rr, cc = draw.circle_perimeter(x+half_size, y+half_size, half_size)
                img[rr, cc] = (0, 255, 0)   
    io.imshow(img)
    io.show()

img, gray_img = test_img = get_test_img()
io.imshow(img)
io.show()
img = resize(img, (300, 300), anti_aliasing=True)
gray_img = resize(gray_img, (300, 300), anti_aliasing=True)
io.imshow(img)
io.show()
print(img.shape)
detect(img, gray_img, clf, selected_feature_coord, selected_feature_type)
