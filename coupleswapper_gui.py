# coding: utf-8
# Form implementation generated from reading ui file 'C:\pyprojects\faceswapper\gui.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

import os,traceback
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore,QtGui, QtWidgets
import cv2
import numpy as np
from coupleswapper import Coupleswapper,TooManyFaces,NoFace

class Ui_Form(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_Form,self).__init__()
        self.swapper=[]
        self.im_path=''
        self.cur_im_path=''
        self.img_swapped=None
        self.img_ori=None
        self.compare=None
        
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(270, 360)
        Form.setAccessibleName("")
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        Form.setWindowIcon(QtGui.QIcon('./male_female.ico'))
        self.help_label=QtWidgets.QLabel(Form)
        self.verticalLayout.addWidget(self.help_label)
        with open('./readme.txt','r',encoding='utf8') as f:
            self.readme=f.read()
        #显示运行log
        self.statu_text=QtWidgets.QTextBrowser(Form)
        self.vb=self.statu_text.verticalScrollBar()
        self.verticalLayout.addWidget(self.statu_text)#支持滚轮
        #加载按钮
        self.bt_load = QtWidgets.QPushButton(Form)
        self.bt_load.setDefault(True)
        self.bt_load.setObjectName("bt_load")
        self.bt_load.clicked.connect(self.load_image)
        self.verticalLayout.addWidget(self.bt_load)
        #转换按钮
        self.bt_swap = QtWidgets.QPushButton(Form)
        self.bt_swap.setDefault(True)
        self.bt_swap.setObjectName("bt_swap")
        self.bt_swap.clicked.connect(self.swap)
        self.verticalLayout.addWidget(self.bt_swap)
        #保存结果按钮
        self.bt_save = QtWidgets.QPushButton(Form)
        self.bt_save.setDefault(True)
        self.bt_save.setObjectName("bt_save")
        self.bt_save.clicked.connect(self.save_result)
        self.verticalLayout.addWidget(self.bt_save)
        #保存对比图按钮
        self.bt_save_comp = QtWidgets.QPushButton(Form)
        self.bt_save_comp.setDefault(True)
        self.bt_save_comp.clicked.connect(self.save_compare)
        self.bt_save_comp.setObjectName("bt_save_comp")
        self.verticalLayout.addWidget(self.bt_save_comp)


        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "交换♂身体"))
        self.bt_load.setText(_translate("Form", "加载图片"))
        self.bt_swap.setText(_translate("Form", "转换"))
        self.bt_save.setText(_translate("Form", "保存结果"))
        self.bt_save_comp.setText(_translate("Form", "保存对比"))
        self.help_label.setText(_translate("Form",self.readme))
        self.statu_text.setText(_translate('Form','欢迎使用，请选择文件'))


    def load_image(self):
        '''
        加载原图
        '''
        try:
            im_path,_=QFileDialog.getOpenFileName(self,'打开图片文件','./','Image Files(*.png *.jpg *.bmp)')
            if not os.path.exists(im_path):
                return
            self.im_path=im_path
            self.statu_text.append('打开图片文件：'+self.im_path)
            if not self.swapper:
                self.swapper=Coupleswapper([self.im_path])
            elif not self.im_path== self.cur_im_path:
                self.swapper.load_heads([self.im_path])
            self.img_ori=self.swapper.heads[os.path.split(self.im_path)[-1]][0]
            cv2.imshow('Origin',self.img_ori)
        except (TooManyFaces,NoFace):
            self.statu_text.append(traceback.format_exc()+'\n人脸定位失败，请重新选择！保证照片中有两张可识别的人脸。')
            return

    def swap(self):
        '''
        执行换脸
        '''
        if not (self.swapper and os.path.exists(self.im_path)):
            return
        self.statu_text.append('转换成功！')
        self.img_swapped=self.swapper.swap(os.path.split(self.im_path)[-1])
        self.img_swapped[self.img_swapped>254.9]=254.9
        self.img_swapped=self.img_swapped.astype('uint8')
        cv2.imshow('Result',self.img_swapped)
        
    def save_result(self):
        '''
        保存结果
        '''
        output_path,_=QFileDialog.getSaveFileName(self,'选择保存位置','./','Image Files(*.png *.jpg *.bmp)')
        if not output_path:
            self.statu_text.append('无效路径,请重新选择')
            return
        self.swapper.save(output_path,self.img_swapped)
        self.statu_text.append('成功保存到：'+output_path)
    
    def save_compare(self):
        '''
        保存对比图
        '''
        self.compare=np.concatenate([self.img_ori,self.img_swapped],1)
        cv2.imshow('Compare',self.compare)
        output_path,_=QFileDialog.getSaveFileName(self,'选择保存位置','./','Image Files(*.png *.jpg *.bmp)')
        if not output_path:
            self.statu_text.append('无效路径,请重新选择')
            return
        self.swapper.save(output_path,self.compare)
        self.statu_text.append('成功保存对比图到：'+output_path)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    try:
        Form.show()
    except:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
    sys.exit(app.exec_())

