import os.path

import cv2
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import numpy as np
import os

def create_dir(us_num):
    os.makedirs('photos/'+str(us_num),exist_ok =True)
def detect_face_from_video(rgb,num_photo,us_num):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    path = 'photos'
    if (len(faces) == 0):
        return -1;
    else:
        create_dir('s'+ str(us_num));
       # cv2.imwrite(os.path.join(path,str(user_name)+str('/')+str(num_photo)+'.jpg'), gray);
        cv2.imwrite('photos/' + 's'+ str(us_num)+ '/' + '1.jpg', gray);
    return 0;
def detect_face(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None


    # under the assumption that there will be only one face,
    # extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]

def objectTracker1():
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    while True:
        # Get a fresh frame
        (depth, _), (rgb, _) = get_depth(), get_video()

        # Build a two panel color image
        d3 = np.dstack((depth, depth, depth)).astype(np.uint8)
        da = np.hstack((d3, rgb))

        #gray1 = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('face.jpg',gray1)
        num_photo = 1;
        us_num = 3

        # Simple Downsample
        cv2.imshow('both', np.array(da[::2, ::2, ::-1]))
        cv2.imshow("3d",depth)
        cv2.waitKey(5)

        rval, frame = vc.read()
        # Capture frame-by-frame
        ret, frame = vc.read()
        frame = rgb
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        objects = faceCascade.detectMultiScale(gray, scaleFactor=1.2,
                                               minNeighbors=1, minSize=(40, 40),
                                              flags=cv2.CASCADE_SCALE_IMAGE)


        # Draw a rectangle around the faces
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("preview", rgb)
        if (detect_face_from_video(rgb, num_photo, us_num) == 0):
            return

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    #vc.release()
    cv2.destroyWindow("preview")

if __name__ == '__main__':
    objectTracker1()