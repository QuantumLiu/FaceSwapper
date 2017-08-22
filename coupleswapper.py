# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 07:56:47 2017

@author: Quantum Liu
"""


import sys,os,traceback
import cv2
import dlib
import numpy as np

from faceswapper import Faceswapper

class TooManyFaces(Exception):
    '''
    定位到太多脸
    '''
    pass


class NoFace(Exception):
    '''
    没脸
    '''
    pass


class Coupleswapper(Faceswapper):
    '''
    双人照人脸交换器类，继承自Faceswapper类
    实例化时载入多个照片资源
    '''
    def get_landmarks(self,im,fname,n=2):
        '''
        人脸定位和特征提取，定位到两张及以上脸或者没有人脸将抛出异常
        im:
            照片的numpy数组
        fname:
            照片名字的字符串
        返回值:
            人脸特征(x,y)坐标的矩阵
        '''
        rects = self.detector(im, 1)
        
        if len(rects) >=5:
            raise TooManyFaces('Too many faces in '+fname)
        if len(rects) <2:
            raise NoFace('No enough face in' +fname)
        return [np.matrix([[p.x, p.y] for p in self.predictor(im, rect).parts()]) for rect in rects]

    
    def swap(self,im_name):
        '''
        主函数 人脸交换
        im_name
            合影图片的键名字符串
        face_path:
            脸来源的图像路径名
        '''
        im,landmarks=self.heads[im_name]
        out_im=im.copy()
        for i in [1,-1]:
            landmarks_head,landmarks_face=landmarks[:2][::i]
            M = self.transformation_from_points(landmarks_head[self.ALIGN_POINTS],
                                           landmarks_face[self.ALIGN_POINTS])
            
            face_mask = self.get_face_mask(im, landmarks_face)
            warped_mask = self.warp_im(face_mask, M, im.shape)
            combined_mask = np.max([self.get_face_mask(im, landmarks_head), warped_mask],
                                      axis=0)
            warped_face = self.warp_im(im, M, im.shape)
            warped_corrected_im = self.correct_colours(im, warped_face, landmarks_head)
            out_im=out_im * (1.0 - combined_mask) + warped_corrected_im * combined_mask
        return out_im
if __name__=='__main__':
    '''
    命令行运行：
    python release.py [合影路径] [输出图片路径](可选，默认./output.jpg)
    '''
    im_path,out=(sys.argv[1] if len(sys.argv)>=2 else 'trump_cp.jpg'),(sys.argv[2] if len(sys.argv)>=3 else 'output.jpg')
    swapper=Coupleswapper([im_path])
    output_im=swapper.swap(os.path.split(im_path)[-1])
    swapper.save(out,output_im)
    output_im[output_im>254.9]=254.9
    cv2.imshow('',output_im.astype('uint8'))
    cv2.waitKey()
