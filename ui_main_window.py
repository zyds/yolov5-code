# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(776, 339)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.input = QLabel(self.centralwidget)
        self.input.setObjectName(u"input")
        self.input.setGeometry(QRect(30, 20, 331, 201))
        self.input.setScaledContents(True)
        self.input.setAlignment(Qt.AlignCenter)
        self.output = QLabel(self.centralwidget)
        self.output.setObjectName(u"output")
        self.output.setGeometry(QRect(420, 20, 331, 201))
        self.output.setScaledContents(True)
        self.output.setAlignment(Qt.AlignCenter)
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(360, 20, 61, 201))
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.det_image = QPushButton(self.centralwidget)
        self.det_image.setObjectName(u"det_image")
        self.det_image.setGeometry(QRect(30, 260, 331, 41))
        self.det_video = QPushButton(self.centralwidget)
        self.det_video.setObjectName(u"det_video")
        self.det_video.setGeometry(QRect(420, 260, 331, 41))
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.input.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u539f\u59cb\u56fe\u7247", None))
        self.output.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u68c0\u6d4b\u7ed3\u679c", None))
        self.det_image.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u7247\u68c0\u6d4b", None))
        self.det_video.setText(QCoreApplication.translate("MainWindow", u"\u89c6\u9891\u68c0\u6d4b", None))
    # retranslateUi

