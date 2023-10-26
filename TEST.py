from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt , QObject
from PyQt5.QtWidgets import QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap,QIcon
from PyQt5.QtCore import Qt , QObject
from PyQt5.QtWidgets import QMessageBox ,QApplication
import os.path
import cv2
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import numpy as np
import os



class Ui_MainWindow(object):


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(670, 420)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(20, 40, 211, 311))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 67, 17))
        self.label.setObjectName("label")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(280, 40, 151, 181))
        self.graphicsView_2.setObjectName("graphicsView_2")
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
        self.RefButton = QtWidgets.QPushButton(self.centralwidget)
        self.RefButton.setGeometry(QtCore.QRect(500, 150, 89, 25))
        self.RefButton.setObjectName("pushButton")
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
        self.pushButton.clicked.connect(lambda: self.Scan_but())
        self.pushButton.clicked.connect(lambda: self.Push_but())


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Photo Rec"))
        self.label_2.setText(_translate("MainWindow", "Photo DB"))
        self.label_3.setText(_translate("MainWindow", "User name "))
        self.label_4.setText(_translate("MainWindow", "User name "))
        self.pushButton.setText(_translate("MainWindow", "Add"))
        self.RefButton.setText(_translate("MainWindow", "Scan"))

    def Scan_but(self):
        af =0
    def Push_but(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("If you want add user , push button <<Add>>")
        msg.setWindowTitle("User undefined")
        msg.exec()

        print("Ck")

    def show_face_from_DB(self,us_num):
        self.scene2 = QtWidgets.QGraphicsScene()
        self.graphicsView_2.setScene(self.scene2)
        self.image_qt2 = QImage('photos/s'+ str(us_num)+ '/ 3.jpg')

        pic2 = QtWidgets.QGraphicsPixmapItem()
        pic2.setPixmap(QPixmap.fromImage(self.image_qt2.scaled(151, 181, Qt.IgnoreAspectRatio, Qt.FastTransformation)))
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
        pic.setPixmap(QPixmap.fromImage(self.image_qt.scaled(211, 311 ,Qt.IgnoreAspectRatio,Qt.FastTransformation)))
        self.scene.setSceneRect(0, 0, 400, 400)
        self.scene.addItem(pic)
    def print_name_us(self,Name_us):
        self.textBrowser.setText(Name_us)

    def MessageBox(self):
        QMessageBox.about("User undefined", "If you want add user , push button <<Add>>")






if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()





    sys.exit(app.exec_())
