
from skimage import io
import cv2
import matplotlib.pyplot as plt
import numpy as np

file = "images/1.jpg"

# ОБУЧЕНИЕ
image = io.imread(file)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#View input image
print("Input image:")
fig, ax = plt.subplots()
ax.imshow(image)
plt.show()
#View input image ends

#Find circles
#
output = image.copy()
dish_img = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1., 150, minRadius=250, maxRadius=480)

dishes = {}
hists = {}

i = 0
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        print("Блюдо", i)
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        cv2.circle(dish_img, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(dish_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        circle_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        cv2.circle(circle_img,(x, y), r, (255,255,255), -1)
        avg_rgb = cv2.mean(image_hsv, mask = circle_img)[::-1]
        
        print("Назовите блюдо:")
        fig, ax = plt.subplots()
        ax.imshow(dish_img)
        plt.show()
        dish_img = image.copy()
        dish = input()
        dishes[dish] = avg_rgb        
        i += 1

    print("Circles:")
    fig, ax = plt.subplots()
    ax.imshow(output)
    plt.show()

print("dishes", dishes)
#Find circles ends

classes = {
        'potato': (0.0, 97.5, 150.4, 103.0), 
        'salad': (0.0, 111.1, 143.4, 105.7), 
        'bread': (0.0, 137.4, 118.4, 107.8), 
        'soup': (0.0, 110.4, 99.5, 75.5), 
        'fruit': (0.0, 186.7, 84.5, 104.8)
}

# ТЕСТИРОВАНИЕ
test_img = io.imread("images/1.jpg")
test_img_hsv = cv2.cvtColor(test_img, cv2.COLOR_RGB2HSV)
#
print("Input image:")
fig, ax = plt.subplots()
ax.imshow(test_img)
plt.show()
#View input image ends

#Find circles

output = test_img.copy()
dish_img = test_img.copy()
gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1., 150, minRadius=250, maxRadius=480)

dishes_new = {}
hists_new = {}

i = 0
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        
        cv2.circle(output, (x, y), r, (255, 0, 0), 14)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        cv2.circle(dish_img, (x, y), r, (255, 0, 0), 14)
        cv2.rectangle(dish_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        circle_img = np.zeros((test_img.shape[0], test_img.shape[1]), np.uint8)
        cv2.circle(circle_img,(x, y), r, (255,255,255), -1)
        avg_rgb = cv2.mean(test_img_hsv, mask = circle_img)[::-1]
        
        fig, ax = plt.subplots()
        ax.imshow(dish_img)
        plt.show()
        dish_img = test_img.copy()
        dishes_new[i] = avg_rgb
        
        key_of_dish = ""
        min_diff = 300
        for key in classes:
            diff = np.abs(np.subtract(classes[key], avg_rgb))
            avg_diff_hsv = np.mean(diff)
#            print(avg_rgb, classes[key], key)
            if avg_diff_hsv < min_diff :
                min_diff = avg_diff_hsv
                key_of_dish = key
        
        print("Блюдо: ", key_of_dish, "{0:.2f}".format(min_diff))
        
        i += 1