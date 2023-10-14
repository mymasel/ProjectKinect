import os.path
import importlib
import time

import cv2
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import numpy as np
import os


# importlib.import_module("OpenCV-Face-Recognition-Python")
def create_dir(us_num):
    os.makedirs('photos/' + str(us_num), exist_ok=True)


def detect_face_from_video(rgb, us_num):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return -1;
    else:
        cv2.imwrite('photo_for_detect/' + str(1) + '.jpg', rgb);
    return 0;


def AddUser( rgb, us_num,i):
    #name = ""
    #input(name)
    #subjects.append(name)
    j= 0
    if (detect_face_from_video(rgb, us_num) == 0):
        for h in range(10000000):
                j +=1
        cv2.imwrite('photos/'+ 's'+ str(us_num)+ '/' + str(i) + '.jpg', rgb);
    return 0;





def objectTracker1():

    us_num =1

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    i = 1
    while True:

        # Get a fresh frame
        (depth, _), (rgb, _) = get_depth(), get_video()

        # Build a two panel color image
        d3 = np.dstack((depth, depth, depth)).astype(np.uint8)
        da = np.hstack((d3, rgb))

        # Simple Downsample

        cv2.waitKey(5)

        # Capture frame-by-frame
        ret, frame = vc.read()
        frame = rgb
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        objects = faceCascade.detectMultiScale(gray, scaleFactor=1.2,
                                               minNeighbors=1, minSize=(40, 40),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
        # Draw a rectangle around the faces
        cv2.imshow("preview", rgb)
        if (detect_face_from_video(rgb, us_num)) == 0 and i!=13:
            key = cv2.waitKey(20)
            AddUser(frame, us_num, i)
            i+=1
            print(i)


        key = cv2.waitKey(20)


        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")


if __name__ == '__main__':
    objectTracker1()