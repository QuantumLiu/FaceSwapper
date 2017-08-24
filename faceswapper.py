# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:10:26 2017

@author: Quantum Liu

reference:https://github.com/matthewearl/faceswap
"""


import sys,os,traceback
import cv2
import dlib
import numpy as np


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


class Faceswapper():
    '''
    人脸交换器类
    实例化时载入多个头照片资源
    '''
    def __init__(self,heads_list=[],predictor_path="./data/shape_predictor_68_face_landmarks.dat"):
        '''
        head_list:
            头（背景和发型）来源图片的路径的字符串列表，根据此列表在实例化时载入多个头像资源，
            并获得面部识别点坐标，以字典形式存储，键名为文件名
        predictor_path:
            dlib资源的路径
        '''
        #五官等标记点
        self.PREDICTOR_PATH = predictor_path
        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))

        # 人脸的完整标记点
        self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                                       self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)
        
        # 来自第二张图（脸）的标记点，眼、眉、鼻子、嘴，这一部分标记点将覆盖第一张图的对应标记点
        self.OVERLAY_POINTS = [self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_BROW_POINTS + self.RIGHT_BROW_POINTS,
            self.NOSE_POINTS + self.MOUTH_POINTS]
        
        # 颜色校正参数
        self.COLOUR_CORRECT_BLUR_FRAC = 0.6
        
        #人脸定位、特征提取器，来自dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)
        
        #头像资源
        self.heads={}
        if heads_list:
            self.load_heads(heads_list)
        
    def load_heads(self,heads_list):
        '''
        根据head_list添加更多头像资源
        '''
        self.heads.update({os.path.split(name)[-1]:(self.read_and_mark(name)) for name in heads_list})

    def get_landmarks(self,im,fname,n=1):
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
        
        if len(rects) > n:
            raise TooManyFaces('No face in '+fname)
        if len(rects) < 0:
            raise NoFace('Too many faces in '+fname)
        return np.matrix([[p.x, p.y] for p in self.predictor(im, rects[0]).parts()])

    def read_im(self,fname,scale=1):
        '''
        读取图片
        '''
# =============================================================================
#         im = cv2.imread(fname, cv2.IMREAD_COLOR)
# =============================================================================
        im = cv2.imdecode(np.fromfile(fname,dtype=np.uint8),-1)
        if type(im)==type(None):
            print(fname)
            raise ValueError('Opencv read image {} error, got None'.format(fname))
        return im

    def read_and_mark(self,fname):
        im=self.read_im(fname)
        return im,self.get_landmarks(im,fname)
    
    def resize(self,im_head,landmarks_head,im_face,landmarks_face):
        '''
        根据头照片和脸照片的大小（分辨率）调整图片大小，增强融合效果
        '''
        scale=np.sqrt((im_head.shape[0]*im_head.shape[1])/(im_face.shape[0]*im_face.shape[1]))
        if scale>1:
            im_head=cv2.resize(im_head,(int(im_head.shape[1]/scale),int(im_head.shape[0]/scale)))
            landmarks_head=(landmarks_head/scale).astype(landmarks_head.dtype)
        else:
            im_face=cv2.resize(im_face,(int(im_face.shape[1]*scale),int(im_face.shape[0]*scale)))
            landmarks_face=(landmarks_face*scale).astype(landmarks_face.dtype)
        return im_head,landmarks_head,im_face,landmarks_face

    def draw_convex_hull(self,im, points, color):
        '''
        勾画多凸边形
        '''
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)
    
    def get_face_mask(self,im, landmarks,ksize=(11,11)):
        '''
        获得面部遮罩
        '''
        mask = np.zeros(im.shape[:2], dtype=np.float64)
    
        for group in self.OVERLAY_POINTS:
            self.draw_convex_hull(mask,
                             landmarks[group],
                             color=1)
    
        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    
        mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        mask = cv2.GaussianBlur(mask, ksize, 0)
    
        return mask
        
    def transformation_from_points(self,points1, points2):
        """
        Return an affine transformation [s * R | T] such that:
    
            sum ||s*R*p1,i + T - p2,i||^2
    
        is minimized.
        计算仿射矩阵
        参考：https://github.com/matthewearl/faceswap/blob/master/faceswap.py
        """
        # Solve the procrustes problem by subtracting centroids, scaling by the
        # standard deviation, and then using the SVD to calculate the rotation. See
        # the following for more details:
        #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)
    
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
    
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2
    
        U, S, Vt = np.linalg.svd(points1.T * points2)
    
        # The R we seek is in fact the transpose of the one given by U * Vt. This
        # is because the above formulation assumes the matrix goes on the right
        # (with row vectors) where as our solution requires the matrix to be on the
        # left (with column vectors).
        R = (U * Vt).T
    
        return np.vstack([np.hstack(((s2 / s1) * R,
                                           c2.T - (s2 / s1) * R * c1.T)),
                             np.matrix([0., 0., 1.])])
    
    def warp_im(self,im, M, dshape):
        '''
        人脸位置仿射变换
        '''
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return output_im
    
    def correct_colours(self,im1, im2, landmarks_head):
        '''
        颜色校正
        '''
        blur_amount = int(self.COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                                  np.mean(landmarks_head[self.LEFT_EYE_POINTS], axis=0) -
                                  np.mean(landmarks_head[self.RIGHT_EYE_POINTS], axis=0)))
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
        return im2.astype(np.float64) *im1_blur.astype(np.float64) /im2_blur.astype(np.float64)
    
    def swap(self,head_name,face_path):
        '''
        主函数 人脸交换
        head_name：
            头资源的键名字符串
        face_path:
            脸来源的图像路径名
        '''
        im_head,landmarks_head,im_face,landmarks_face=self.resize(*self.heads[head_name],*self.read_and_mark(face_path))
        M = self.transformation_from_points(landmarks_head[self.ALIGN_POINTS],
                                       landmarks_face[self.ALIGN_POINTS])
        
        face_mask = self.get_face_mask(im_face, landmarks_face)
        warped_mask = self.warp_im(face_mask, M, im_head.shape)
        combined_mask = np.max([self.get_face_mask(im_head, landmarks_head), warped_mask],
                                  axis=0)
        
        warped_face = self.warp_im(im_face, M, im_head.shape)
        warped_corrected_im2 = self.correct_colours(im_head, warped_face, landmarks_head)
        
        out=im_head * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        return out
    
    def save(self,output_path,output_im):
        '''
        保存图片
        '''
        cv2.imencode('.jpg',output_im)[1].tofile(output_path)
# =============================================================================
#         cv2.imwrite(os.path.abspath(output_path.encode('utf-8').decode('gbk')), output_im)
# =============================================================================

if __name__=='__main__':
    '''
    命令行运行：
    python faceswapper.py <头路径> <脸路径> <输出图片路径>(可选，默认./output.jpg)
    '''
    head,face_path,out=sys.argv[1],sys.argv[2],(sys.argv[3] if len(sys.argv)>=4 else 'output.jpg')
    swapper=Faceswapper([head])
    output_im=swapper.swap(os.path.split(head)[-1],face_path)#返回的numpy数组
    swapper.save(out,output_im)
    output_im[output_im>254.9]=254.9
    cv2.imshow('',output_im.astype('uint8'))
    cv2.waitKey()

