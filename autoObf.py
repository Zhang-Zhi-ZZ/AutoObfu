# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '771.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QApplication,QLabel
import sys
import skimage
from skimage.viewer import ImageViewer
import imagehash
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter,ImageDraw
import random
import os,sys
from prediction import predictor

class Ui_Dialog(object):
    global imgName, oImg, resultImg, pred, hashed
    def setupUi(self, Dialog):
        global pred
        pred = predictor()
        Dialog.setObjectName("Dialog")
        Dialog.resize(330, 481)

        self.open_file = QtWidgets.QPushButton(Dialog)
        self.open_file.clicked.connect(self.getImage)
        self.open_file.setGeometry(QtCore.QRect(170, 70, 91, 31))
        self.open_file.setObjectName("open_file")


        # button for obfuscating methods 
        self.ECB = QtWidgets.QPushButton(Dialog)
        self.ECB.clicked.connect(self.fingerprint)
        self.ECB.setGeometry(QtCore.QRect(180, 150, 89, 32))
        self.ECB.setObjectName("fingerprint")

        self.AES = QtWidgets.QPushButton(Dialog)
        self.AES.clicked.connect(self.encrypt)
        self.AES.setGeometry(QtCore.QRect(90, 150, 89, 32))
        self.AES.setObjectName("encrypt")

        self.Gaussian = QtWidgets.QPushButton(Dialog)
        self.Gaussian.clicked.connect(self.blur)
        self.Gaussian.setGeometry(QtCore.QRect(0, 150, 89, 32))
        self.Gaussian.setObjectName("blur")


        # Words 
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 80, 181, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(10, 120, 221, 16))
        self.label_2.setObjectName("label_2")

        # Photos 
        self.originalImg = QtWidgets.QLabel(Dialog)
        self.originalImg.setGeometry(QtCore.QRect(10, 270, 151, 151))
        self.originalImg.setObjectName("originalImg")

        self.result = QtWidgets.QLabel(Dialog)
        self.result.setGeometry(QtCore.QRect(170, 270, 151, 151))
        self.result.setObjectName("result")


        # button for Screen and Classify
        self.initialScrern = QtWidgets.QPushButton(Dialog)
        self.initialScrern.clicked.connect(self.screen)
        self.initialScrern.setGeometry(QtCore.QRect(10, 200, 81, 32))
        self.initialScrern.setObjectName("Screen")
        self.classify = QtWidgets.QPushButton(Dialog)
        self.classify.clicked.connect(self.classification)
        self.classify.setGeometry(QtCore.QRect(180, 200, 81, 32))
        self.classify.setObjectName("Classify")

        # Words 
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(10, 240, 91, 20))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(180, 240, 91, 20))
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(110, 25, 50, 30))
        self.label_6.setObjectName("label_6")


        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.open_file.setText(_translate("Dialog", "Open"))
        self.ECB.setText(_translate("Dialog", "Fingerprint"))
        self.AES.setText(_translate("Dialog", "Encryption"))
        self.Gaussian.setText(_translate("Dialog", "Blurring"))
        self.initialScrern.setText(_translate("Dialog", "Screen"))
        self.classify.setText(_translate("Dialog", "Classify"))
        self.label.setText(_translate("Dialog", "Please select your photo:"))
        self.label_2.setText(_translate("Dialog", "Available obfuscating methods:"))
        self.label_3.setText(_translate("Dialog", "Original Image:"))
        self.label_4.setText(_translate("Dialog", "Result:"))
        self.label_6.setText(_translate("Dialog", "AutoObfu"))


    def getImage(self):
        global imgName
        oImg = QFileDialog().getOpenFileName(None,"Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        imgName = oImg[0]
        jpg = QtGui.QPixmap(imgName).scaled(150, 150)
        self.originalImg.setPixmap(jpg)
    

    def encrypt(self):
        global resultImg
        resultImg = 1
        tmp = imgName
        image = cv2.imread(tmp)  # reads the image
        pixels = image
        row = 28
        col = 28
        row1 = 1000003
        phi = [0 for x1 in range(row1)]
        occ = [0 for x1 in range(row1)]
        primes = [] 
        phi[1] = 1
        for i in range(2,1000001):
            if(phi[i] == 0):
                phi[i] = i-1
                primes.append(i)
                for j in range (2*i,1000001,i):
                    if(occ[j] == 0):
                        occ[j] = 1
                        phi[j] = j
                    phi[j] = (phi[j]*(i-1))//i
        p = primes[random.randrange(1,167)]
        q = primes[random.randrange(1,167)]
        n = p*q
        mod = n
        phin1 = phi[n]
        phin2 = phi[phin1]
        e = primes[random.randrange(1,9000)]
        mod1 = phin1
        def power1(x,y,m):
            ans=1
            while(y>0):
                if(y%2==1):
                    ans=(ans*x)%m
                y=y//2
                x=(x*x)%m
            return ans
        d = power1(e,phin2-1,mod1)
        enc = [[0 for x in range(row)] for y in range(col)]
        dec = [[0 for x in range(row)] for y in range(col)]
        for i in range(col):
            for j in range(row):
                r = pixels[i][j]
                r1 = power1(r+10,e,mod)
                enc[i][j] = r1
        img = np.array(enc,dtype = np.uint8)
        img1 = Image.fromarray(img)
        iName= 'Encrypted.png'
        img1.save(iName)
        blurred = QtGui.QPixmap(iName).scaled(150,150)
        self.result.setPixmap(blurred)
        self.result.show()


    def fingerprint(self):
        global resultImg, hashed
        resultImg = 2
        tmp = imgName
        hashed = imagehash.phash(Image.open(tmp))
        img = Image.new('RGB', (150,150), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10,50), "Fingerprint: \n"+str(hashed), fill=(255, 255, 0)) 
        iName= 'Fingerprint.png'
        img.save(iName)
        blurred = QtGui.QPixmap(iName).scaled(150,150)
        self.result.setPixmap(blurred)
        self.result.show()


    def blur(self):
        global resultImg
        resultImg = 0
        tmp = imgName
        image = cv2.imread(tmp)  # reads the image
        new_image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
        iName= 'Blurred.png'
        cv2.imwrite(iName, new_image)
        blurred = QtGui.QPixmap(iName).scaled(150,150)
        self.result.setPixmap(blurred)
        self.result.show()


    def screen(self):
        img = Image.new('RGB', (150,150), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10,50), "Recommendation: \n Blur method", fill=(255, 255, 0)) 
        iName= 'screen.png'
        img.save(iName)
        screen = QtGui.QPixmap(iName).scaled(150,150)
        self.result.setPixmap(screen)


    def temp(self, word):
        img = Image.new('RGB', (150,150), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10,50),"The predicted result:\n" + word, fill=(255, 255, 0)) 
        iName= 'temp.png'
        img.save(iName)
        screen = QtGui.QPixmap(iName).scaled(150,150)
        self.result.setPixmap(screen)


    def classification(self):
        global resultImg, pred, hashed
        try:
            if resultImg == 0:
                self.temp(pred.pred_blur())
            elif resultImg == 1:
                self.temp(pred.pred_encrypt())
            elif resultImg == 2:
                self.temp(pred.pred_fingerprint(hashed))
                print("call the fingerprint to classify")

        except Exception as e:
            print("no photo has been upload")
            raise e


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())