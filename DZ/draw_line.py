import numpy as np
import cv2 as cv

cap = cv.VideoCapture('videos/optima_VID_20190514_222920(0).mp4')

# Define the codec and create VideoWriter object
#fourcc = cv.VideoWriter_fourcc(*'XVID')
#out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))

fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('fish.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

print('cap', cap)
print('cap.get(3)', cap.get(3))
print('cap.get(4)', cap.get(4))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
#        frame = cv.flip(frame,0)

        cv.line(frame, (530, 0), (530, int(cap.get(4))), (0, 0, 0), 12)
        # write the flipped frame
        out.write(frame)

        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()