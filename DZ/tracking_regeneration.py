import numpy as np
import cv2 as cv
import math  
from scipy.spatial.distance import euclidean

def distance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist
    
def get_rects(p):
    threshold = 150
    rects = []
    p_tmp = [val for sublist in p for val in sublist]
    point_added  = False
    for point in p_tmp:
        if len(rects) == 0:
            rects.append([point])
        else:
            for rect in rects:
                if point_added:
                    break
                for rect_point in rect:
                    if distance(point[0], point[1], rect_point[0], rect_point[1]) < threshold:
                        rect.append(point)
                        point_added = True
                        break
            if point_added == False:
                rects.append([point])
        point_added = False
    return rects
        
    
def centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return centroid
    
def geometric_mediod(points):
    distance = euclidean
    geometric_mediod = min(map(lambda p1:(p1,sum(map(lambda p2:distance(p1,p2),points))),points), key = lambda x:x[1])[0]
    return geometric_mediod


def is_point_changed(prev_distance, points, centr):
    threshold = 50
    for d, p in zip(prev_distance, points):
        print('is changed', d, distance(centr[0], centr[1], p[0], p[1]), abs(d - distance(centr[0], centr[1], p[0], p[1])))
        if abs(d - distance(centr[0], centr[1], p[0], p[1])) > threshold:
            return True
    return False
    
cap = cv.VideoCapture('fish.avi')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read() # Захватывает, декодирует и возвращает следующий видеокадр
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params) #Определяет сильные углы на изображении
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('fish_regeneration_process.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

rects = []
centr = 0
distances = []

rects = get_rects(p0)
p0 = np.array(rects[0]).reshape(-1,1,2)
centr = centroid(rects[0])
for p in rects[0]:
    distances.append(distance(centr[0], centr[1], p[0], p[1]))

while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    
    rects = get_rects(p0)
    p0 = np.array(rects[0]).reshape(-1,1,2)
    
    if is_point_changed(distances, rects[0], centr):
        mask1 = np.zeros_like(old_gray)
        mask1[:] = 255
        cv.rectangle(mask1,  (int(centr[0])-120, int(centr[1])-60), (int(centr[0])+120, int(centr[1])+60), 0)
        p0 = cv.goodFeaturesToTrack(old_gray, mask = mask1, **feature_params)
        rects = get_rects(p0)
        p0 = np.array(rects[0]).reshape(-1,1,2)
    
   
    centr = centroid(rects[0]) 
    cv.rectangle(frame, (int(centr[0])-120, int(centr[1])-60), (int(centr[0])+120, int(centr[1])+60), (0, 255, 0), 2)


    # calculate optical flow
    # Вычисляет оптический поток для разреженного набора объектов, 
    # используя итерационный метод Лукаса-Канаде с пирамидами
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel() #  непрерывный плоский массив.
        c,d = old.ravel()
#        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        frame = cv.circle(frame,(a,b),5,(0, 255, 0),-1)
        
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    out.write(img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv.destroyAllWindows()
cap.release()
out.release()
