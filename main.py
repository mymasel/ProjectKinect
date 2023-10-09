import os.path
import importlib
import cv2
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import numpy as np
import os
importlib.import_module("OpenCV-Face-Recognition-Python")
def create_dir(us_num):
    os.makedirs('photos/'+str(us_num),exist_ok =True)
def detect_face_from_video(rgb,num_photo,us_num):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return -1;
    else:
        create_dir(us_num)
        for i in range(0,12):
            for h in range(0,10000):
                cv2.imwrite('photos/' + 's'+ str(us_num)+ '/' + str(i)+'.jpg', rgb);
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

def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containin images for current subject 
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            #cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

def AddUser(subjects):
    name = ""
    input(name)
    subjects.append(name)


def predict(test_img):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)

    # predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    # get name of respective label returned by face recognizer
    label_text = subjects[label]

    return img
    
def objectTracker1():
    print("Preparing data...")
    faces, labels = prepare_training_data("photos")
    print("Data prepared")

    # print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    num_photo = 1
    us_num = 1
    (depth, _), (rgb, _) = get_depth(), get_video()
    subjects = [""]
    print("Add? 1/0")
    quest = 0
    if (input(quest) == 1):
        AddUser(subjects)
        detect_face_from_video(rgb, num_photo, us_num)

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    while True:
        # Get a fresh frame

        # Build a two panel color image
        d3 = np.dstack((depth, depth, depth)).astype(np.uint8)
        da = np.hstack((d3, rgb))

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
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    #vc.release()
    cv2.destroyWindow("preview")

if __name__ == '__main__':
    objectTracker1()