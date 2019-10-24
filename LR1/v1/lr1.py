import os
from skimage import data, io, filters, data_dir, color
import numpy as np
import math  
from scipy import stats


class LR1():
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.trainAppleFolder = os.path.join(dir_path,  'Training/Apple Red Yellow/')
        self.trainPapayaFolder = os.path.join(dir_path, 'Training/Papaya/')
        self.testAppleFolder = os.path.join(dir_path,   'Testing/Apple Red Yellow/')
        self.testPapayaFolder = os.path.join(dir_path,  'Testing/Papaya/')
        
    def load_images_from_folder(self, folder):
        images = []
        for filename in os.listdir(folder):
            img = io.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
        return images
    
    def loadTrainApple(self):
        return self.load_images_from_folder(self.trainAppleFolder)
    
    def loadTrainPapaya(self):
        return self.load_images_from_folder(self.trainPapayaFolder)
    
    def loadTestApple(self):
        return self.load_images_from_folder(self.testAppleFolder)
    
    def loadTestPapaya(self):
        return self.load_images_from_folder(self.testPapayaFolder)
    
    def getOneTestApple(self):
        return self.load_images_from_folder(self.testAppleFolder)[10]
    def getOneTestPapaya(self):
        return self.load_images_from_folder(self.testPapayaFolder)[10]
    
    def scharr(self, image):
        image = color.rgb2gray(image)
        edges = filters.scharr(image)
        return edges
    
    def imshow(self, image):
        io.imshow(image)
        io.show()
        
    def getFruitHeight(self, image):
        rowLength = image.shape[0]
        colLength = image.shape[1]
        middleColIndex = int(colLength / 2)
        start = 0
        end = 0
        i = 0
        while start == 0 and i < rowLength:
            if image[i][middleColIndex] != 0:
                start = i
            i += 1
        i = rowLength - 1 
        while end == 0 and i >= 0:
            if image[i][middleColIndex] != 0:
                end = i
            i -= 1
        return end - start
    
    def getFruitWidth(self, image):
        rowLength = image.shape[0]
        colLength = image.shape[1]
        middleRowIndex = int(rowLength / 2)
        row = image[middleRowIndex]
        start = 0
        end = 0
        i = 0
        while start == 0 and i < colLength:
            if row[i] != 0:
                start = i
            i += 1
        i = colLength - 1 
        while end == 0 and i >= 0:
            if row[i] != 0:
                end = i
            i -= 1
        return end - start
    
#    def getFruitPerimeter(self, image):
#        rowLength = image.shape[0]
#        colLength = image.shape[1]
#        kontur = []
#        for i in range(0, rowLength):
#            point = 0
#            j=0
#            row = image[i]
#            while point == 0 and j < colLength:
#                if row[j] != 0:
#                    point = j
#                j += 1
#            if point != 0:
#                kontur.append(row[point])
#
#        for i in range(rowLength,0,-1):
#            i -= 1
#            j=colLength - 1
#            point=0
#            row = image[i]
#            while point == 0 and j >= 0:
#                if row[j] != 0:
#                    point = j
#                j -= 1
#            if point != 0:
#                kontur.append(row[point])
##        print(kontur)
#        return len(kontur)
    
    def perimV(self, image):            # сканирование сверху вниз и обратно
        rowLength = image.shape[0]
        colLength = image.shape[1]
        kontur = []
        middleColIndex =  colLength / 2
        
        for i in range(0, rowLength):   # сверху вниз
            point = 0
            j=0
            row = image[i]
            while point == 0 and j < colLength: # слево направо
                if row[j] != 0:
                    point = j
                j += 1
            if point != 0 and j <= middleColIndex and (len(kontur) == 0 or abs(kontur[-1][1] - j) < 20):
#            if point != 0 and j <= middleColIndex:
#            if point != 0:
                kontur.append((i,j))

        for i in range(rowLength,0,-1):     # снизу вверх
            i -= 1
            j=colLength - 1
            point=0
            row = image[i]
            while point == 0 and j >= 0:    # справо налево
                if row[j] != 0:
                    point = j
                j -= 1
            if point != 0 and j >= middleColIndex and (len(kontur) == 0 or abs(kontur[-1][1] - j) < 20):
#            if point != 0 and j >= middleColIndex:
#            if point != 0:
                kontur.append((i,j))
#        print(kontur)
        distance = 0
        prev = ()
        for k in kontur:
            if prev:
                distance += self.distance(prev, k)
            prev = k
        distance += self.distance(kontur[-1], kontur[0])
        return distance
    
    def perimH(self, image):            #сканирование слево направо и обратно
        rowLength = image.shape[0]
        colLength = image.shape[1]
        kontur = []
        middleRowIndex =  rowLength / 2
        for i in range(0, colLength):       # слево направо
            point = 0
            j=0
            while point == 0 and j < rowLength:     # сверху вниз
                if image[j][i] != 0:
                    point = j
                j += 1
#            if point != 0:
            if point != 0 and j <= middleRowIndex and (len(kontur) == 0 or abs(kontur[-1][1] - j) < 20):
                kontur.append((j,i))
        for i in range(colLength,0,-1):     # справо налево
            i -= 1
            j=rowLength - 1
            point=0
            while point == 0 and j >= 0:    # снизу вверх
                if image[j][i] != 0:
                    point = j
                j -= 1
#            if point != 0:
            if point != 0 and j >= middleRowIndex and (len(kontur) == 0 or abs(kontur[-1][1] - j) < 20):
                kontur.append((j,i))
        distance = 0
        prev = ()
        for k in kontur:
            if prev:
                distance += self.distance(prev, k)
            prev = k
        distance += self.distance(kontur[-1], kontur[0])
        return distance

    def distance(self, a, b):
        return math.sqrt(((b[0] - a[0])**2) + ((b[1] - a[1])**2))
        
        
        
    def arrayInfo(self, nparray):
        print(nparray.ndim)  #the number of axes (dimensions) of the array.
        print(nparray.shape)
        print(nparray.size)
        print(nparray.dtype)
        print(nparray.itemsize)
        print(nparray.data)
        print(nparray.sum())
        print(nparray.min())
        print(nparray.max())
        print(nparray.mean())
        print(np.median(nparray))
        print(nparray.ravel())
        print(nparray)
        
class model():
    h = 0
    w = 0
    p = 0
    classes = {}
    meanClasses = {}
    eps = {
                "h": 10,
                "w": 10,
                "p": 15,
        }
    
    def __init__(self, LR1):
        self.lr = LR1
        
    def train(self, images, label):
        for image in images:
            self.addToTrain( self.getFeature(image), label)
            
    def addToTrain(self, features, label):
        if not self.classes.get(label):
            self.classes[label] = {
                "h":  [features[0]],
                "w":  [features[1]],
                "p": [features[2]],
                "n":  1
                }
        else:
            self.classes[label]["h"].append(features[0])
            self.classes[label]["w"].append(features[1])
            self.classes[label]["p"].append(features[2])
            self.classes[label]["n"] += 1
            
    def getTrainedClasses(self):
        self.meanClasses = {}
        for key, value in self.classes.items():
#            n = value["n"]
#            h = value["h"] / n
#            w = value["w"] / n
#            p = stats.mode(value["p"])
            p = np.median(value["p"])
            h = np.median(value["h"])
            w = np.median(value["w"])
            self.meanClasses[key] = {
                    "h":  h,
                    "w":  w,
                    "p": p,
                    }
        return self.meanClasses
    
    def predictOne(self, image):
        features = self.getFeature(image)
#        print(features)
        for key, value in self.getTrainedClasses().items():
            h = value["h"]
            w = value["w"]
            p = value["p"]
            dh = abs(h - features[0])
            dw = abs(w - features[1])
            dp = abs(p - features[2])
#            print(key, dh, dw, dp)
            if dh < self.eps["h"] and dw < self.eps["w"]: #and dp < self.eps["p"]:
                return key
        return "empty"
    
    def getFeature(self, image):
        lr = self.lr
        image = lr.scharr(image)
        image = (image > 0.05).astype(int)
        h = lr.getFruitHeight(image)
        w = lr.getFruitWidth(image)
        p = 0
        if w > h:
            tmp = w
            w = h
            h = tmp
            p = lr.perimH(image)
        else:
            p = lr.perimV(image)
#        p = lr.perimV(image)
#        print([h,w,p])
        return [h,w,p]
        
    
    
    
    def predict(self, images, label):
        testImagesCount = len(images)
        print("test ", label, " images count: ", testImagesCount)
        lr = self.lr
        hit = 0
        for image in images:
            im1 = lr.scharr(image)
            im1 = (im1 > 0.05).astype(int)
#            lr.imshow(im1)
            prediction = self.predictOne(image)
            if prediction == label:
                hit += 1
        print("hit: ", hit)
        hitPercent = (hit / testImagesCount) * 100
        print(hitPercent, "% \n")
        
        
        
    
    
    
