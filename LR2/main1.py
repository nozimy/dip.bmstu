import cv2
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

nemo = cv2.imread('./images/6vtzGFrQpak.jpg')
plt.imshow(nemo)
plt.show()

nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
plt.imshow(nemo)
plt.show()

#=== 3D RGB
#r, g, b = cv2.split(nemo)
#fig = plt.figure()
#axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
#
#axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
#axis.set_xlabel("Red")
#axis.set_ylabel("Green")
#axis.set_zlabel("Blue")
#plt.show()
#=== 3D

hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
plt.imshow(hsv_nemo)
plt.show()
#=== 3D HSV
h, s, v = cv2.split(hsv_nemo)
fig = plt.figure()
axis = fig.add_subplot(111, projection="3d")


print(pixel_colors)

axis.scatter(h.flatten()[:10000], s.flatten()[:10000], v.flatten()[:10000], marker=".")
#axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()


#=== 3D
