from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt , QObject
from PyQt5.QtWidgets import QMessageBox
import os.path
import cv2
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import numpy as np
import os



class Ui_MainWindow(object):


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(670, 417)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(20, 40, 211, 311))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 67, 17))
        self.label.setObjectName("label")
        self.graphicsView2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView2.setGeometry(QtCore.QRect(280, 40, 151, 181))
        self.graphicsView2.setObjectName("graphicsView_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(280, 20, 67, 17))
        self.label_2.setObjectName("label_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(280, 260, 151, 21))
        self.textBrowser.setObjectName("textBrowser")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(280, 230, 81, 17))
        self.label_3.setObjectName("label_3")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(500, 60, 151, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(500, 40, 81, 17))
        self.label_4.setObjectName("label_4")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(500, 100, 89, 25))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 670, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.Push_but()


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Photo Rec"))
        self.label_2.setText(_translate("MainWindow", "Photo DB"))
        self.label_3.setText(_translate("MainWindow", "User name "))
        self.label_4.setText(_translate("MainWindow", "User name "))
        self.pushButton.setText(_translate("MainWindow", "Add"))

    def Push_but(self):
        if self.lineEdit.text() == "":
            return
        self.pushButton.clicked.connect(lambda: ui.Add_button(self,rgb))


    #'photos/s' + str(us_num) + '/ 3.jpg'
    def show_face_from_DB(self,us_num):
        self.scene2= QtWidgets.QGraphicsScene()
        self.graphicsView2.setScene(self.scene2)
        self.image_qt2 = QImage('photos/s' + str(us_num)+ '/3.jpg')

        pic2 = QtWidgets.QGraphicsPixmapItem()
        pic2.setPixmap(QPixmap.fromImage(self.image_qt2.scaled(141, 171, Qt.IgnoreAspectRatio, Qt.FastTransformation)))
        self.scene2.setSceneRect(0, 0, 400, 400)
        self.scene2.addItem(pic2)


    def show_face_for_detect(self):

        #image =QImage(str(1)+'.jpg')
        #pixmap = QPixmap()
        #self.graphicsView.
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.image_qt = QImage('photo_for_detect/1.jpg')

        pic = QtWidgets.QGraphicsPixmapItem()
        pic.setPixmap(QPixmap.fromImage(self.image_qt.scaled(201, 301 ,Qt.IgnoreAspectRatio,Qt.FastTransformation)))
        self.scene.setSceneRect(0, 0, 400, 400)
        self.scene.addItem(pic)
    def print_name_us(self,Name_us):
        self.textBrowser.setText(Name_us)
    def Add_button(self,rgb):
        text = self.lineEdit.text()
        print(1)
        for h in range(1, 13):
            AddUser(rgb, len(subjects)+1, h,text)

        f = open('subjects', 'w')
        for i in range(len(subjects)):
            f.write(subjects[i])
        f.close()

    def MessageBox(self):
        QMessageBox.about("User undefined", "If you want add user , push button <<Add>>")


def create_dir(us_num):
    os.makedirs('photos/'+str(us_num),exist_ok =True)

def detect_face_from_video(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return -1;
    else:
        cv2.imwrite('photo_for_detect/1.jpg', rgb);
    return 0;

def AddUser( rgb, us_num,i,text):
    create_dir(us_num)
    subjects.append(text)

    if (detect_face_from_video(rgb) == 0):
        cv2.imwrite('photos/'+ 's'+ str(us_num)+ '/' + str(i) + '.jpg', rgb);
        update_frame()
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

        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containin images for current subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)


        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image)

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



print("Preparing data...")
faces, labels = prepare_training_data("photos")
print("Data prepared")
subjects = [" "]
f = open('subjects.txt', 'r+' )
for item in  f:
    subjects += item.split(",")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Get a fresh frame
(depth, _), (rgb, _) = get_depth(), get_video()
# Build a two panel color image
d3 = np.dstack((depth, depth, depth)).astype(np.uint8)
da = np.hstack((d3, rgb))

# Simple Downsample

def predict(test_img):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)
    if rect.all() == None:
        return None
    # predict the image using our face recognizer
    label, confidence = face_recognizer.predict(face)
    if (label == 0):
        return None
    # get name of respective label returned by face recognizer
    label_text = subjects[label]
    return label


def update_frame():
    (depth, _), (rgb, _) = get_depth(), get_video()
    # Build a two panel color image
    d3 = np.dstack((depth, depth, depth)).astype(np.uint8)
    da = np.hstack((d3, rgb))

    # Simple Downsample
    cv2.waitKey(5)

    # Capture frame-by-frame
    frame = rgb
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.waitKey(40)


def objectTracker1():
    num_photo = 1
    us_num = 5
    while True:
        # Get a fresh frame
        (depth, _), (rgb, _) = get_depth(), get_video()
        # Build a two panel color image
        d3 = np.dstack((depth, depth, depth)).astype(np.uint8)
        da = np.hstack((d3, rgb))

        # Simple Downsample
        cv2.waitKey(5)

        # Capture frame-by-frame

        frame = rgb
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = faceCascade.detectMultiScale(gray, scaleFactor=1.2,
                                               minNeighbors=1, minSize=(40, 40),
                                               flags=cv2.CASCADE_SCALE_IMAGE)

        # Draw a rectangle around the faces

        if (detect_face_from_video(rgb)) == 0 :
           tvtv = 'photo_for_detect/' + str(1) + '.jpg'
           test_img = cv2.imread(tvtv)
           label = predict(test_img)
           if (label != None):
               return subjects[label] , label
           else:
               return None

        key = cv2.waitKey(20)
        if key == 27:
            f = open('subjects', 'w')
            f.write(subjects)
            f.close()   # exit on ESC
            break





if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    Name_us, num_lable = objectTracker1()
    if Name_us == None :
            ui.show_face_for_detect()
            ui.MessageBox()
    else:
            ui.show_face_for_detect()
            ui.show_face_from_DB(num_lable)
            ui.print_name_us(Name_us)

    sys.exit(app.exec_())
